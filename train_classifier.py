import torch
import datasets
import transformers
import time
import os
import re
import shutil
import tyro
from tqdm import tqdm
from collections import defaultdict

import contextlib
import training_utils
import classifier_lib
from datetime import datetime
from dataclasses import dataclass
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
import loggers


def readable_str(number):
    """Format a number to thousands with k suffix."""
    return f"{number/1000:.1f}k"


def remove_file_or_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def safe_save(save_func, save_path):
    tmp_path = save_path + ".tmp"
    remove_file_or_dir(tmp_path)
    remove_file_or_dir(save_path)
    save_func(tmp_path)
    os.rename(src=tmp_path, dst=save_path)


@dataclass
class Args:
    debug: bool = False
    track: bool = False
    logger_tags: list[str] | None = None
    run_name: str | None = None

    data_path: str = "VGS-AI/OpenR1-VM"
    validation_data_path: str | None = None  # if not set, will use data_path
    dataset_name: str | None = None
    data_num_response: int = 8
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    compile: bool = False
    # required for long sequences
    gradient_checkpointing: bool = True
    seed: int = 1337
    micro_batch_size: int = 4
    # 128 * 16k ~ 2M tokens per batch. Matches GPT3 paper.
    total_batch_size: int = 128
    max_length: int = 16384
    disable_lr_schedule: bool = False
    max_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_norm_clip: float = 5.0

    p_dropout: float = 0.05
    use_dora: bool = False

    num_steps: int = 4000
    eval_size: int = 128  # 128 * 16 = 2048
    eval_every: int = 200
    save_every: int = 200
    max_keep_ckpts: int = 20
    small_group_size: int = 2
    push_to_hub: bool = False

    num_labels: int = 3  # 1 or 3
    label_key: str = "labels"
    train_bt_model: bool = False


    def __post_init__(self):
        # save every must be multiple of eval every
        assert self.small_group_size <= self.data_num_response, f"small_group_size ({self.small_group_size}) must be less than or equal to data_num_response ({self.data_num_response})"
        assert self.data_num_response % self.small_group_size == 0, f"data_num_response ({self.data_num_response}) must be multiple of small_group_size ({self.small_group_size})"

        current_date = datetime.now()
        self.run_date = current_date.strftime('%Y-%m-%d').replace('-', '_')
        if self.run_name is None:
            if self.dataset_name is not None:
                data_str = f"{self.dataset_name}_"
            else:
                data_str = ""
            self.run_name = f"{data_str}lr_{self.max_lr}_bs_{self.total_batch_size}_steps_{self.num_steps}_seed_{self.seed}"

        scratch_dir = os.environ.get("SCRATCH", "scratch_dir")
        self.save_dir = os.path.join(scratch_dir, "qsharp_ckpts", self.run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print("save_dir:", self.save_dir)

        if self.validation_data_path is None:
            self.validation_data_path = self.data_path


def main(args: Args, ddp: bool, device: str, ddp_local_rank: int, ddp_rank: int, ddp_world_size: int, master_process: bool):
    assert torch.cuda.is_available()
    device_type = 'cuda'
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    print(f"Initializing model...")
    model_loading_kwargs = dict(attn_implementation="flash_attention_2", torch_dtype=dtype, use_cache=False, attention_dropout=args.p_dropout)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    logger_run_id = None
    latest_ckpt_step = None
    optimizer_state_dict = None
    optimizer_save_path = os.path.join(args.save_dir, "optimizer.pt")
    if os.path.exists(args.save_dir) and os.path.exists(optimizer_save_path):
        optimizer_state_dict = torch.load(optimizer_save_path, weights_only=True)
        logger_run_id = optimizer_state_dict["logger_run_id"]
        latest_ckpt_step = optimizer_state_dict["step"]
        optimizer_state_dict = optimizer_state_dict["optimizer_state_dict"]

    if latest_ckpt_step is not None:
        latest_ckpt_path = os.path.join(args.save_dir, f"model_{latest_ckpt_step}")
        print(f"Got latest step {latest_ckpt_step}! Loading model from {latest_ckpt_path} and optimizer from {optimizer_save_path}...")
        classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(latest_ckpt_path, **model_loading_kwargs)
    else:
        classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(args.model_path, **model_loading_kwargs, num_labels=args.num_labels)
        if args.train_bt_model:
            classifier.train_bt_model = True
    if args.gradient_checkpointing:
        classifier.gradient_checkpointing_enable()
    classifier.to(device)
    if ddp:
        classifier = DistributedDataParallel(
            classifier,
            device_ids=[ddp_local_rank],
            find_unused_parameters=True,  # otherwise throws error..
            static_graph=True,  # maybe disable if no_sync() https://github.com/pytorch/pytorch/issues/143580
        )
    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        print("Compiling model...")
        t0 = time.time()
        classifier = torch.compile(classifier)
        print(f"Compilation took {time.time() - t0:.2f}s")

    # setup optimizer
    classifier.train()
    raw_classifier = classifier.module if ddp else classifier
    optimizer = training_utils.configure_optimizer(
        raw_classifier, lr=args.max_lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay, device_type=device_type)
    if optimizer_state_dict is not None:
        print("Loading optimizer state...")
        optimizer.load_state_dict(optimizer_state_dict)
    warmup_iters = max(round(0.05 * args.num_steps), 100)
    min_lr = 0.1 * args.max_lr

    # setup dataset and loaders
    print("Loading dataset...")
    grouped_keys = ["roll_outs_ids", args.label_key, "roll_in_ids"]
    train_dataset = datasets.load_dataset(args.data_path, split='train')

    print("Flattening datasets...")
    flattened_train_dataset = training_utils.FlattenedDataset(train_dataset,
            grouped_keys=grouped_keys, shared_keys=[], big_group_size=args.data_num_response, small_group_size=args.small_group_size)

    print("Constructing dataloaders...")
    grad_accum_steps = args.total_batch_size // (args.micro_batch_size * ddp_world_size)
    # assume max_length is a power of 2
    assert args.max_length % 16 == 0, "max_length must be a multiple of 16"
    pad_multiple = args.max_length // 16
    collate_fn = training_utils.RollInOutCollator(tokenizer, roll_in_key="roll_in_ids", roll_out_key="roll_outs_ids", max_length=args.max_length, pad_multiple=pad_multiple)
    train_sampler = training_utils.EndlessSampler(flattened_train_dataset, batch_size=args.micro_batch_size, process_rank=ddp_rank, num_processes=ddp_world_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(flattened_train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=4, prefetch_factor=8)
    train_loader = iter(train_loader)

    if args.eval_every > 0:
        val_dataset = datasets.load_dataset(args.validation_data_path, split='validation').select(range(args.eval_size))
        print("Loaded validation dataset with size", len(val_dataset))
        flattened_val_dataset = training_utils.FlattenedDataset(val_dataset,
                grouped_keys=grouped_keys, shared_keys=[], big_group_size=args.data_num_response, small_group_size=args.small_group_size)
        val_sampler = training_utils.EndlessSampler(flattened_val_dataset, batch_size=args.micro_batch_size, process_rank=ddp_rank, num_processes=ddp_world_size, shuffle=False, endless=False)
        val_loader = torch.utils.data.DataLoader(flattened_val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=4, prefetch_factor=8)

    # initialize tracker
    if args.track and master_process:
        print("Tracking run on neptune...")
        import random
        api_token = os.environ.get("NEPTUNE_API_KEY")
        new_run_id = str(random.randint(1000000000, 9999999999))
        logger_kwargs = dict(api_token=api_token, run_id=new_run_id, project="cornell-rl/q-sharp")
        if logger_run_id is None:
            logger_kwargs.update({
                 "experiment_name": args.run_name,
                 "config": vars(args),
                 "tags": args.logger_tags,
            })
            # also save a json file of args
            import json
            args_json_path = os.path.join(args.save_dir, "args.json")
            with open(args_json_path, "w") as f:
                json.dump(vars(args), f, indent=2)
        else:
            logger_kwargs.update({
                "experiment_name": f"fork@{latest_ckpt_step}-" + args.run_name,
                "fork_run_id": logger_run_id,
                "fork_step": latest_ckpt_step,
            })
        logger_kwargs = {k: v for k, v in logger_kwargs.items() if v is not None}
        run = loggers.NeptuneScaleLogger(**logger_kwargs)
    else:
        run = loggers.DummyLogger()

    # start training loop!
    begin_step = 1 if latest_ckpt_step is None else latest_ckpt_step + 1
    if begin_step > 1:
        print(f"Replaying data to step {begin_step}")
        bar = tqdm(range(1, begin_step)) if master_process else range(1, begin_step)
        for step in bar:
            for _ in range(grad_accum_steps):
                train_loader._next_index()
            if master_process:
                bar.set_description(f"epoch {train_sampler.epoch:2d} | data pos {train_sampler.current_position:6d} | step {step:4d}")

    print(f"{ddp_local_rank=}, {ddp_rank=} starting training loop!")
    for step in range(begin_step, args.num_steps+1):
        last_step = step == args.num_steps

        t0 = time.time()
        optimizer.zero_grad()
        tokens_processed = torch.tensor(0, device=device, dtype=torch.long)
        loss_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
        max_seqlen = torch.tensor(0, device=device, dtype=torch.long)
        avg_seqlen = torch.tensor(0, device=device, dtype=torch.float32)
        valid_tokens_processed = torch.tensor(0, device=device, dtype=torch.long)
        loss_tokens_processed = torch.tensor(0, device=device, dtype=torch.long)
        for micro_step in range(grad_accum_steps):
            batch = next(train_loader)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch[args.label_key]
            loss_mask = batch['roll_out_mask']
            # only all_reduce grads on last step
            context_manager = classifier.no_sync if ddp and step > begin_step and micro_step < grad_accum_steps - 1 else contextlib.nullcontext
            with context_manager():
                with torch.autocast(device_type=device_type, dtype=dtype):
                    outputs = classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loss_mask=loss_mask)
                    loss = outputs.loss / grad_accum_steps

            loss.backward()
            loss_accum += loss.detach()
            cur_seqlen = torch.tensor(batch['input_ids'].shape[1], device=device, dtype=torch.long)
            avg_seqlen += (cur_seqlen / grad_accum_steps)
            max_seqlen = max(max_seqlen, cur_seqlen)
            tokens_processed += (batch['input_ids'].shape[0] * cur_seqlen)
            valid_tokens_processed += attention_mask.sum()
            loss_tokens_processed += loss_mask.sum()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_seqlen, op=dist.ReduceOp.AVG)
            dist.all_reduce(max_seqlen, op=dist.ReduceOp.MAX)
            dist.all_reduce(tokens_processed, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_tokens_processed, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tokens_processed, op=dist.ReduceOp.SUM)

        norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_norm_clip)
        if args.disable_lr_schedule:
            lr = args.max_lr
        else:
            lr = training_utils.get_lr(step, max_lr=args.max_lr, warmup_iters=warmup_iters, lr_decay_iters=args.num_steps, min_lr=min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        optimizer.step()
        if device_type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = tokens_processed / dt
        valid_tokens_per_sec = valid_tokens_processed / dt
        loss_tokens_per_sec = loss_tokens_processed / dt
        if master_process:
            loss_accum = loss_accum.item()
            stats = {
                "epoch": train_sampler.epoch,
                "train/loss": loss_accum,
                "train/lr": lr,
                "train/norm": norm,
                "train/data/avg_seqlen": round(avg_seqlen.item()),
                "train/data/max_seqlen": max_seqlen.item(),
                "train/perf/dt": dt,
                "train/perf/tokens_per_sec": tokens_per_sec,
                "train/perf/valid_tokens_per_sec": valid_tokens_per_sec,
                "train/perf/loss_tokens_per_sec": loss_tokens_per_sec,
                "train/perf/loss_tokens_processed": loss_tokens_processed,
            }
            print(f"ep {stats['epoch']:2d} | step {step:4d} | loss: {loss_accum:.2e} | lr: {lr:.2e} | norm: {norm:.2e} | "
                f"dt: {dt:.2f}s | tok/sec: {readable_str(tokens_per_sec)} | valid tok/sec: {readable_str(valid_tokens_per_sec)} | loss tok/sec: {readable_str(loss_tokens_per_sec)}")
            run.log_metrics(stats, step=step)

        # Evaluation and Save
        with torch.no_grad():
            if args.eval_every > 0 and (step % args.eval_every == 0 or last_step):
                num_eval_batches = len(val_sampler)
                if master_process:
                    print(f"-----------Evaluating at step {step} for {num_eval_batches} batches of {args.total_batch_size}-----------")

                classifier.eval()
                # somehow compiled model raises a nasty error
                raw_classifier = classifier._orig_mod if args.compile else classifier

                t0 = time.time()
                tokens_processed = torch.tensor(0, device=device, dtype=torch.long)
                eval_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                all_preds = []
                all_labels = []
                all_preds_per_pos = defaultdict(list)
                all_labels_per_pos = defaultdict(list)
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch[args.label_key]
                    loss_mask = batch['roll_out_mask']
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        outputs = raw_classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loss_mask=loss_mask)

                    eval_loss += outputs.loss.detach() / num_eval_batches
                    cur_seqlen = torch.tensor(batch['input_ids'].shape[1], device=device, dtype=torch.long)
                    tokens_processed += batch['input_ids'].shape[0] * cur_seqlen

                    all_preds.append(torch.sigmoid(outputs.logits[loss_mask.bool()]))
                    all_labels.append(labels.unsqueeze(-1).expand_as(outputs.logits)[loss_mask.bool()])

                    for pos in range(16):
                        start_pos = 1024 * pos
                        end_pos = 1024 * (pos + 1)
                        loss_mask_per_pos = training_utils.create_pos_mask(loss_mask, start_pos, end_pos)
                        all_preds_per_pos[pos].append(torch.sigmoid(outputs.logits[loss_mask_per_pos.bool()]))
                        all_labels_per_pos[pos].append(labels.unsqueeze(-1).expand_as(outputs.logits)[loss_mask_per_pos.bool()])

                all_preds = torch.cat(all_preds, dim=0)  #  (N,) where N is N_val * N_response * num_loss_toks_per_response
                all_labels = torch.cat(all_labels, dim=0)
                local_stats = training_utils.DistributedMetrics.local_compute(all_preds, all_labels)
                if ddp:
                    dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(tokens_processed, op=dist.ReduceOp.SUM)
                    for key in local_stats:
                        dist.all_reduce(local_stats[key], op=dist.ReduceOp.SUM)
                full_seq_metrics = training_utils.DistributedMetrics.global_combine(local_stats)

                # also log per position metrics: 1024 * k for k in [1, 2, ..., 16]
                per_pos_metrics = {}
                for pos in range(16):
                    all_preds_cur_pos = torch.cat(all_preds_per_pos[pos], dim=0)
                    all_labels_cur_pos = torch.cat(all_labels_per_pos[pos], dim=0)
                    local_stats = training_utils.DistributedMetrics.local_compute(all_preds_cur_pos, all_labels_cur_pos)
                    if ddp:
                        for key in local_stats:
                            dist.all_reduce(local_stats[key], op=dist.ReduceOp.SUM)
                    per_pos_metrics[f"{pos}k_{pos+1}k"] = training_utils.DistributedMetrics.global_combine(local_stats)
                del all_preds, all_labels

                dt = time.time() - t0
                tokens_per_sec = tokens_processed / dt
                if master_process:
                    metrics = {
                        "val/loss": eval_loss,
                        "val/perf/dt": dt,
                        "val/perf/tokens_per_sec": tokens_per_sec,
                    }
                    metrics.update({f"val/{k}": v for k, v in full_seq_metrics.items()})
                    print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

                    for pos_k, v in per_pos_metrics.items():
                        metrics.update({f"val/{pos_k}/{kk}": vv for kk, vv in v.items()})
                    run.log_metrics(metrics, step=step)

                classifier.train()
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if ddp:
                    torch.distributed.barrier()  # wait for all processes to finish eval


            if args.save_every > 0 and (step % args.save_every == 0 or last_step):
                if master_process:
                    classifier.eval()
                    raw_classifier = classifier._orig_mod if args.compile else classifier
                    save_path = os.path.join(args.save_dir, f"model_{step}")
                    raw_classifier = raw_classifier.module if ddp else raw_classifier
                    safe_save(raw_classifier.save_pretrained, save_path)
                    print(f"Saved model to {save_path}")

                    # also save optimizer state
                    save_optimizer_func = lambda path: torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "logger_run_id": run.id,
                    }, path)
                    safe_save(save_optimizer_func, optimizer_save_path)
                    print(f"Saved optimizer state to {optimizer_save_path}")

                    # delete the model 5 checkpoints ago
                    ckpts_sorted_by_step = [ckpt for ckpt in os.listdir(args.save_dir) if re.match(r"^model_\d+$", ckpt)]
                    ckpts_sorted_by_step = sorted(ckpts_sorted_by_step, key=lambda x: int(x.split("_")[-1]))
                    print("Found checkpoints", ckpts_sorted_by_step)
                    if args.max_keep_ckpts > 0 and len(ckpts_sorted_by_step) > args.max_keep_ckpts:
                        for old_ckpt in ckpts_sorted_by_step[:-args.max_keep_ckpts]:
                            print(f"Removing old checkpoint {old_ckpt}")
                            remove_file_or_dir(os.path.join(args.save_dir, old_ckpt))
                            print(f"Removed old checkpoint {old_ckpt}")

                    if args.push_to_hub:
                        raw_classifier.push_to_hub(args.run_name + f"-step-{step}")

                    classifier.train()

                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if ddp:
                    torch.distributed.barrier()  # wait for all processes to finish eval

    if ddp:
        destroy_process_group()

    if args.track:
        run.close()


if __name__ == "__main__":
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        print(f"GPU {ddp_local_rank=} | RANK {ddp_rank=} | WORLD_SIZE {ddp_world_size=}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda'

    args = tyro.cli(Args)
    main(args, ddp, device, ddp_local_rank, ddp_rank, ddp_world_size, master_process)