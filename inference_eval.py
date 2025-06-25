import os
import classifier_lib
import torch
import transformers
import ujson as json
import time
from tqdm import tqdm
from dataclasses import dataclass
import tyro
import deepseek_utils
import benchmark_data
import accuracy_utils
import eval_helpers
import math
import numpy as np
import copy
import sglang as sgl


def get_token_ids(list_of_token_strs, tokenizer):
    ids = tokenizer.batch_encode_plus(list_of_token_strs, add_special_tokens=False)['input_ids']
    out = []
    for x in ids:
        if len(x) == 1:
            out.append(x[0])
        else:
            out.append(None)
    return out


def repeat(xs, n):
    out = []
    for i in range(len(xs)):
        out.extend([xs[i]] * n)
    return out


def unflatten(xs, n):
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def unflatten_variable_n(xs, ns):
    out = []
    idx = 0
    for n in ns:
        out.append(xs[idx:idx+n])
        idx += n
    assert idx == len(xs)
    return out


@dataclass
class Args:
    benchmark: str = "aime-24"
    dataset_size: int = -1  # will be set automatically based on the dataset
    attention_impl: str = "flash_attention_2"
    seed: int = 1337
    piref_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    use_sglang_server: bool = False
    sglang_port: int = 30000
    classifier_ckpt_path: str = "VGS-AI/DeepSeek-VM-1.5B"
    output_path: str = "inference_outputs.jsonl"
    search_type: str = "beam2"
    # fraction of memory to allocate to piref inference engine
    piref_gpu_util: float = 0.5

    batch_size: int = 32
    max_length: int = 16384
    block_size: int = 4096
    num_blocks: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    num_repetitions: int = 1  # number of times to repeat the generation for each input

    # log to neptune. assumes NEPTUNE_API_KEY is set
    neptune_project: str = "FILL_IN_WITH_YOUR_PROJECT_NAME_HERE"


    def __post_init__(self):
        print("classifier_ckpt_path: ", self.classifier_ckpt_path)
        print("output_path: ", self.output_path)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # assert num_repetitions is a power of 2
        assert self.num_repetitions > 0, f"num_repetitions must be positive, but got {self.num_repetitions}."
        assert self.num_repetitions & (self.num_repetitions - 1) == 0, f"num_repetitions must be a power of 2, but got {self.num_repetitions}."


def get_eos_token_id(tokenizer):
    eos_token_id = tokenizer.convert_tokens_to_ids(["<｜end▁of▁sentence｜>"])[0]
    return eos_token_id


@torch.no_grad()
def maybe_finish_generate_and_score(
    piref_model, classifier_model, prompt_ids: list[list[int]], generation_ids: list[list[list[int]]],
    max_length: int, temperature: float, top_p: float, stop_token_ids: list[int], device_type, dtype,
):
    """If generation_ids don't end with stop_token_ids or aren't at max_length, we generate until they do.
    Return the completed generation_ids and their scores.
    """
    num_prompts = len(prompt_ids)
    assert len(generation_ids) == num_prompts, f"generation_ids must have {num_prompts=} elements, but got {len(generation_ids)=}"
    already_completed_indices = []

    num_responses = len(generation_ids[0])
    infer_input_ids = []
    max_num_tokens_list = []
    for i in range(num_prompts):
        assert len(generation_ids[i]) == num_responses, f"{len(generation_ids[i])=} != {num_responses=}"
        for j in range(num_responses):
            if generation_ids[i][j][-1] in stop_token_ids or len(generation_ids[i][j]) >= max_length:
                already_completed_indices.append((i, j))
            else:
                infer_input_ids.append(prompt_ids[i] + generation_ids[i][j])
                max_num_tokens_list.append(max_length - len(generation_ids[i][j]))

    if len(infer_input_ids) > 0:
        sampling_params = [
            dict(
                temperature=temperature,
                top_p=top_p,
                skip_special_tokens=False,
                stop_token_ids=stop_token_ids,
                max_new_tokens=max_num_tokens_list[i],
            ) for i in range(len(infer_input_ids))
        ]
        infer_outputs = piref_model.generate(input_ids=infer_input_ids, sampling_params=sampling_params)
        infer_output_ids = [x['output_ids'] for x in infer_outputs]
        del infer_outputs

        k = 0
        # create a copy to avoid modifying the original generation_ids
        new_generation_ids = copy.deepcopy(generation_ids)
        for i in range(num_prompts):
            for j in range(num_responses):
                if (i, j) not in already_completed_indices:
                    new_generation_ids[i][j].extend(infer_output_ids[k])
                    k += 1
        generation_ids = new_generation_ids
        assert k == len(infer_input_ids), f"Expected {len(infer_input_ids)} outputs, but got {k}."

    scores = score_all_generations(classifier_model, prompt_ids, generation_ids, device_type, dtype)
    return generation_ids, scores


def score_all_generations(classifier_model, prompt_ids, generation_ids, device_type, dtype):
     # now score each generation
    input_output_ids = []
    num_prompts = len(prompt_ids)
    num_responses = len(generation_ids[0])
    for i in range(num_prompts):
        assert len(generation_ids[i]) == num_responses
        for j in range(num_responses):
            input_output_ids.append(prompt_ids[i] + generation_ids[i][j])

    # since we give lots of memory to sgl, the classifier might oom here. thus, we for loop to save memory
    scores_list = []
    device = classifier_model.device
    for i in tqdm(range(num_prompts * num_responses), desc="Scoring generations"):
        input_ids = torch.tensor(input_output_ids[i], device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
        with torch.autocast(device_type=device_type, dtype=dtype):
            classifier_outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            scores = classifier_outputs.success_probs[:, -1].item()
        scores_list.append(scores)
    scores = torch.tensor(scores_list).view(num_prompts, num_responses).tolist()
    return scores


@torch.no_grad()
def generate_beam(
    piref_model, tokenizer, classifier_model, prompt_ids: list[list[int]],
    max_length: int, temperature: float, top_p: float, num_blocks: int, block_size: int, beam_size: int,
    stop_token_ids: list[int] = None, seed: int = 1337,
) -> list[list[int]]:
    """Output all the kept beams for each prompt. The number of kept beams is num_blocks // beam_size.
    Also return the scores for each kept beam.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = 'cuda:0'
    device_type = 'cuda'
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed)

    if stop_token_ids is None:
        stop_token_ids = [get_eos_token_id(tokenizer)]

    assert num_blocks % beam_size == 0, f"num_blocks {num_blocks} must be a multiple of beam_size {beam_size}"
    assert max_length % block_size == 0, f"max_length {max_length} must be a multiple of block_size {block_size}"
    beam_keep_size = num_blocks // beam_size
    batch_size = len(prompt_ids)
    # should be current_batch_size x beam_keep_size
    generated_ids = [[list() for _ in range(beam_keep_size)] for _ in range(batch_size)]
    num_beams_left = batch_size * beam_keep_size  # number of responses left to generate
    beam_left_indices = {i: list(range(beam_keep_size)) for i in range(batch_size)}
    bar = tqdm(total=num_beams_left)
    infer_engine_dt = infer_classifier_dt = postprocess_dt = 0

    while num_beams_left > 0:
        t0 = time.time()
        assert all(len(xs) == beam_keep_size for xs in generated_ids), (
            f"generated_ids must have {beam_keep_size=} elements per batch item, but got {[len(xs) for xs in generated_ids]}"
        )

        flattened_partial_response_lengths = []
        flattened_prompt_partial_responses = []
        for i in range(batch_size):
            for j in beam_left_indices[i]:
                for _ in range(beam_size):
                    flattened_prompt_partial_responses.append(prompt_ids[i] + generated_ids[i][j])
                    flattened_partial_response_lengths.append(len(generated_ids[i][j]))

        sampling_params = dict(
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=False,
        )
        # compute continuation_ids
        max_lengths = [min(block_size, max_length - flattened_partial_response_lengths[i]) for i in range(len(flattened_prompt_partial_responses))]
        sampling_params["stop_token_ids"] = stop_token_ids
        sampling_params_list = [
            {"max_new_tokens": max_lengths[i], **sampling_params} for i in range(len(flattened_prompt_partial_responses))
        ]
        infer_outputs = piref_model.generate(input_ids=flattened_prompt_partial_responses, sampling_params=sampling_params_list)
        continuation_ids = [x['output_ids'] for x in infer_outputs]
        infer_engine_dt += time.time() - t0

        # score all the continuations. note that while kv cache is modified, the classifier also truncates it so it shouldn't be modified.
        t0 = time.time()
        assert len(flattened_prompt_partial_responses) == len(continuation_ids), f"{len(flattened_prompt_partial_responses)} != {len(continuation_ids)}"
        updated_flattened_prompt_partial_responses = [
            flattened_prompt_partial_responses[i] + continuation_ids[i] for i in range(len(flattened_prompt_partial_responses))
        ]
        classifier_scores_list = []
        with torch.autocast(device_type=device_type, dtype=dtype):
            for i in range(len(updated_flattened_prompt_partial_responses)):
                input_ids = torch.tensor(updated_flattened_prompt_partial_responses[i], device=device, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                classifier_outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
                classifier_scores = classifier_outputs.success_probs[0, -1].item()  # last seqlen index
                classifier_scores_list.append(classifier_scores)

        new_generated_ids = []
        current_score_idx = 0
        for i in range(batch_size):
            if len(beam_left_indices[i]) == 0:
                new_generated_ids.append(generated_ids[i])  # no beams left, keep the original ones
                continue

            next_score_idx = current_score_idx + beam_size * len(beam_left_indices[i])
            current_scores = classifier_scores_list[current_score_idx: next_score_idx]
            # top_left_indices = torch.randperm(len(current_scores))[:len(beam_left_indices[i])]
            top_values, top_left_indices = torch.tensor(current_scores).topk(k=len(beam_left_indices[i]), dim=0, sorted=True)
            assert len(top_left_indices) == len(beam_left_indices[i])
            top_indices = [beam_left_indices[i][topk_i // beam_size] for topk_i in top_left_indices]  # subset of beam_left_indices
            # print(f"{i=}, {beam_left_indices[i]=}")
            # print(f"{current_scores=}, {top_left_indices=}, {top_indices=}")
            active_generated_ids = [
                generated_ids[i][top_indices[j]] + continuation_ids[current_score_idx + top_left_indices[j]]
                # generated_ids[i][top_indices[j]] + [7, 35045, 10039, 220, 31947, 35045, 10039, 8] + continuation_ids[current_score_idx + top_left_indices[j]]
                for j in range(len(beam_left_indices[i]))
            ]
            # interleave with done ones to preserve beam_left_indices
            cur_generated_ids = []
            active_idx = 0
            for j in range(beam_keep_size):
                if j in beam_left_indices[i]:
                    cur_generated_ids.append(active_generated_ids[active_idx])
                    active_idx += 1
                else:
                    cur_generated_ids.append(generated_ids[i][j])
            assert active_idx == len(active_generated_ids)

            new_generated_ids.append(cur_generated_ids)
            current_score_idx = next_score_idx

        generated_ids = new_generated_ids

        del continuation_ids, classifier_outputs, classifier_scores
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        infer_classifier_dt += time.time() - t0

        t0 = time.time()
        old_num_beams_left = num_beams_left
        for i in range(batch_size):
            not_done_beam_left_indices = []
            for j in beam_left_indices[i]:
                # print(i, j)
                current_gen_ids = generated_ids[i][j]
                is_done = current_gen_ids[-1] in stop_token_ids  # done if ends with a stop token
                is_trunc = len(current_gen_ids) >= max_length  # truncate when response exceeds max_length tokens
                if not is_done and not is_trunc:
                    not_done_beam_left_indices.append(j)
                else:
                    num_beams_left -= 1
            beam_left_indices[i] = not_done_beam_left_indices

        postprocess_dt += time.time() - t0
        bar.update(old_num_beams_left-num_beams_left)
        bar.set_description(f"Queries left: {num_beams_left:3d}, sgl: {infer_engine_dt:.2f}s, classifier: {infer_classifier_dt:.2f}s, postprocess: {postprocess_dt:.2f}s")
    bar.close()

    # finally score the generated_ids
    generated_scores = score_all_generations(classifier_model, prompt_ids, generated_ids, device_type, dtype)
    return {"response_ids": generated_ids, "response_scores": generated_scores}


def contiguous_view(nparr, shape):
    """Create a contiguous view of a numpy array with the given shape."""
    if nparr.flags['C_CONTIGUOUS']:
        return nparr.reshape(shape)
    else:
        raise ValueError("Input array is not C-contiguous. Please ensure the input is a contiguous numpy array.")


def get_logger(logger_kwargs):
    print("logger kwargs: ", logger_kwargs)
    import neptune_scale

    tags = logger_kwargs.pop('tags')
    group_tag = tags.pop("group_tag")
    experiment = logger_kwargs.pop('experiment')
    experiment_name = logger_kwargs.pop('name')
    logger_kwargs.update({
        'api_token': os.environ.get("NEPTUNE_API_KEY"),
        'project': experiment,
        'run_id': experiment_name,
        'experiment_name': experiment_name,
    })
    print("Starting to log to Neptune...")
    run = neptune_scale.Run(**logger_kwargs)
    run.add_tags([group_tag], group_tags=True)
    run.log_configs(tags)
    return run


def get_tag(args, dvts_n=1):
    tag = f"{args.search_type}_sz_{args.block_size}_rep_{args.num_repetitions}_temp_{args.temperature}"
    if dvts_n == 1:
        return tag
    else:
        assert dvts_n > 1
        return f"{tag}_dvtsn_{dvts_n}"


def log_to_server(args: Args):
    data = eval_helpers.load_jsonl(args.output_path)
    assert len(data) == (args.dataset_size * args.num_repetitions), f"Expected {args.dataset_size * args.num_repetitions} items in output, but got {len(data)}. Check if the output file is complete."
    tag = get_tag(args, dvts_n=1)
    tags = {
        "search": str(args.search_type),
        "block_size": str(args.block_size),
        "seed": str(args.seed),
        "group_tag": str(tag),
        "dvts_n": str(1),
    }
    experiment_name = f"{tag}_seed_{args.seed}"
    logger_kwargs = dict(
        experiment=args.neptune_project,
        name=experiment_name,
        tags=tags,
    )
    run = get_logger(logger_kwargs)

    scores = np.array([d['generated_scores'] for d in data])
    rewards = np.array([d['reward'] for d in data])
    processed_answers = np.array([d['processed_answer'] for d in data])
    num_beams = rewards.shape[1]
    bon_rewards = torch.tensor([
        eval_helpers.classifier_bon(rewards=rewards[i], classifier_values=scores[i], n=num_beams)
        for i in range(len(data))
    ]).float()
    # num_repetitions is outer loop
    bon_rewards = bon_rewards.view(args.num_repetitions, args.dataset_size).transpose(0, 1)  # repetitions is the outer loop, so need to transpose.
    bon_m, bon_ci = eval_helpers.estimate_mean_and_error(bon_rewards)

    # compute weighted majority vote for each repetition
    if num_beams == 1:
        maj_m, maj_ci = bon_m, bon_ci  # if num_beams == 1, bon and maj are the same
        wmaj_m, wmaj_ci = bon_m, bon_ci
        pass_m, pass_ci = bon_m, bon_ci
    else:
        maj_outputs = [
            eval_helpers.weighted_majority(rewards=rewards[i], gen_answers=processed_answers[i], n=num_beams, weights=scores[i])
            for i in range(len(data))
        ]
        maj_rewards = torch.tensor([x['maj_rewards'] for x in maj_outputs]).float()
        maj_rewards = maj_rewards.view(args.num_repetitions, args.dataset_size).transpose(0, 1)  # [dataset_size, num_repetitions]
        maj_m, maj_ci = eval_helpers.estimate_mean_and_error(maj_rewards)

        wmaj_rewards = torch.tensor([x['wmaj_rewards'] for x in maj_outputs]).float()
        wmaj_rewards = wmaj_rewards.view(args.num_repetitions, args.dataset_size).transpose(0, 1)  # [dataset_size, num_repetitions]
        wmaj_m, wmaj_ci = eval_helpers.estimate_mean_and_error(wmaj_rewards)

        pass_rewards = torch.tensor([rewards[i].max() for i in range(len(data))]).float()
        pass_rewards = pass_rewards.view(args.num_repetitions, args.dataset_size).transpose(0, 1)  # [dataset_size, num_repetitions]
        pass_m, pass_ci = eval_helpers.estimate_mean_and_error(pass_rewards)


    scores_per_prompt = contiguous_view(scores, (args.num_repetitions, args.dataset_size, num_beams)).transpose(1, 0, 2)  # [num_repetitions, dataset_size, num_beams]
    scores_per_prompt = np.ascontiguousarray(scores_per_prompt)  # ensure it's contiguous
    rewards_per_prompt = contiguous_view(rewards, (args.num_repetitions, args.dataset_size, num_beams)).transpose(1, 0, 2)  # [num_repetitions, dataset_size, num_beams]
    rewards_per_prompt = np.ascontiguousarray(rewards_per_prompt)  # ensure it's contiguous
    processed_answers_per_prompt = contiguous_view(processed_answers, (args.num_repetitions, args.dataset_size, num_beams)).transpose(1, 0, 2)  # [num_repetitions, dataset_size]
    processed_answers_per_prompt = np.ascontiguousarray(processed_answers_per_prompt)  # ensure it's contiguous

    print("Logging bon and wmaj metrics...")
    run.log_metrics({
        f'{args.benchmark}/bon_m': bon_m,
        f'{args.benchmark}/bon_ci': bon_ci,
        f'{args.benchmark}/maj_m': maj_m,
        f'{args.benchmark}/maj_ci': maj_ci,
        f'{args.benchmark}/wmaj_m': wmaj_m,
        f'{args.benchmark}/wmaj_ci': wmaj_ci,
        f'{args.benchmark}/pass_m': pass_m,
        f'{args.benchmark}/pass_ci': pass_ci,
    }, step=args.num_blocks)
    run.close()

    # now for every power of 2 until num_repetitions, compute the DVTS with weighted majority vote:
    if args.num_repetitions > 1:
        for i in range(1, int(math.log2(args.num_repetitions)) + 1):
            dvts_n = 2 ** i

            tag = get_tag(args, dvts_n=dvts_n)
            tags = {
                "search": str(args.search_type),
                "block_size": str(args.block_size),
                "seed": str(args.seed),
                "group_tag": str(tag),
                "dvts_n": str(dvts_n),
            }
            experiment_name = f"{tag}_seed_{args.seed}"
            logger_kwargs = dict(
                experiment=args.neptune_project,
                name=experiment_name,
                tags=tags,
            )
            run = get_logger(logger_kwargs)

            dvts_num_beams = num_beams * dvts_n  # number of beams in the DVTS
            dvts_num_repetitions = args.num_repetitions // dvts_n
            dvts_scores = contiguous_view(scores_per_prompt, (args.dataset_size * dvts_num_repetitions, dvts_num_beams))  # [dataset_size * dvts_num_repetitions, dvts_num_beams]
            dvts_rewards = contiguous_view(rewards_per_prompt, (args.dataset_size * dvts_num_repetitions, dvts_num_beams))
            dvts_processed_answers = contiguous_view(processed_answers_per_prompt, (args.dataset_size * dvts_num_repetitions, dvts_num_beams))  # [dataset_size * dvts_num_repetitions, dvts_num_beams]

            # compute dvts_bon and dvts_wmaj
            dvts_bon_rewards = np.array([
                eval_helpers.classifier_bon(dvts_rewards[i], classifier_values=dvts_scores[i], n=dvts_num_beams)
                for i in range(args.dataset_size * dvts_num_repetitions)
            ])
            dvts_bon_rewards = contiguous_view(dvts_bon_rewards, (args.dataset_size, dvts_num_repetitions))  # [dataset_size, dvts_num_repetitions]
            dvts_bon_m, dvts_bon_ci = eval_helpers.estimate_mean_and_error(dvts_bon_rewards)

            dvts_maj_outputs = [
                eval_helpers.weighted_majority(
                    rewards=dvts_rewards[i], gen_answers=dvts_processed_answers[i], n=dvts_num_beams, weights=dvts_scores[i]
                ) for i in range(args.dataset_size * dvts_num_repetitions)
            ]
            dvts_maj_rewards = np.array([x['maj_rewards'] for x in dvts_maj_outputs])
            dvts_maj_rewards = contiguous_view(dvts_maj_rewards, (args.dataset_size, dvts_num_repetitions))
            dvts_maj_m, dvts_maj_ci = eval_helpers.estimate_mean_and_error(dvts_maj_rewards)

            dvts_wmaj_rewards = np.array([x['wmaj_rewards'] for x in dvts_maj_outputs])
            dvts_wmaj_rewards = contiguous_view(dvts_wmaj_rewards, (args.dataset_size, dvts_num_repetitions))
            dvts_wmaj_m, dvts_wmaj_ci = eval_helpers.estimate_mean_and_error(dvts_wmaj_rewards)

            dvts_pass_rewards = np.array([dvts_rewards[i].max() for i in range(args.dataset_size * dvts_num_repetitions)])
            dvts_pass_rewards = contiguous_view(dvts_pass_rewards, (args.dataset_size, dvts_num_repetitions))
            dvts_pass_m, dvts_pass_ci = eval_helpers.estimate_mean_and_error(dvts_pass_rewards)

            # log to num_blocks * dvts_n, since that's the compute budget
            print(f"Logging dvts_{dvts_n}...")
            run.log_metrics({
                f'{args.benchmark}/bon_m': dvts_bon_m,
                f'{args.benchmark}/bon_ci': dvts_bon_ci,
                f'{args.benchmark}/maj_m': dvts_maj_m,
                f'{args.benchmark}/maj_ci': dvts_maj_ci,
                f'{args.benchmark}/wmaj_m': dvts_wmaj_m,
                f'{args.benchmark}/wmaj_ci': dvts_wmaj_ci,
                f'{args.benchmark}/pass_m': dvts_pass_m,
                f'{args.benchmark}/pass_ci': dvts_pass_ci,
            }, step=args.num_blocks*dvts_n)
            run.close()

    print("Finished everything. Bye bye!")


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, num_repeats):
        self.base_dataset = base_dataset
        self.num_repeats = num_repeats
        self.total_len = len(base_dataset) * num_repeats

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_dataset)
        repeat_idx = idx // len(self.base_dataset)
        item = self.base_dataset[base_idx]
        base_idx = idx % len(self.base_dataset)

        assert isinstance(item, dict), f"Expected dict from base_dataset, got {type(item)}"
        item['repeat_idx'] = repeat_idx
        item['data_idx'] = base_idx

        return item


def math_verify_wrapper(args):
    return accuracy_utils.math_verify_check(*args)


@torch.no_grad()
def main(args: Args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = 'cuda:0'
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision('high')

    # load dataset
    dataset = benchmark_data.get_dataset(args.benchmark)
    args.dataset_size = len(dataset)

    # load existing outputs
    skip_indices = []
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            n_done = 0
            for line in f:
                d = json.loads(line)
                global_idx = d["repeat_idx"] * args.dataset_size + d["data_idx"]
                skip_indices.append(global_idx)
                n_done += 1

        assert set(skip_indices) == set(range(n_done)), f"Skip indices {skip_indices} do not match expected range 0 to {n_done-1}."
        print(f"Found {n_done} existing samples in {args.output_path}.")

        if len(skip_indices) == (args.dataset_size * args.num_repetitions):
            print(f"Output file {args.output_path} already contains all samples.")
            log_to_server(args)
            exit(0)
        print(f"Already done {skip_indices=}")
    else:
        with open(args.output_path, "w") as f:
            f.write("")


    extra_kwargs = {}
    piref_model = sgl.Engine(
        model_path=args.piref_model,
        dtype=dtype,
        mem_fraction_static=args.piref_gpu_util,
        random_seed=args.seed,
        skip_tokenizer_init=True,
        **extra_kwargs,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.piref_model, padding_side="left")
    model_loading_kwargs = dict(attn_implementation=args.attention_impl, torch_dtype=dtype, use_cache=True)
    classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(args.classifier_ckpt_path, **model_loading_kwargs).to(device)
    classifier.eval()
    print("Finished loading piref and classifier.")

    # has problem and answer columns
    def preprocess(example):
        formatted_problem = deepseek_utils.format_roll_in(example["problem"])
        input_ids = tokenizer(formatted_problem, add_special_tokens=False)["input_ids"]
        example["input_ids"] = input_ids
        return example

    def simple_collate(batch):
        # simply return a list of input_ids, problems and answers
        batch = [preprocess(x) for x in batch]
        return {
            "input_ids": [x["input_ids"] for x in batch],
            "problem": [x["problem"] for x in batch],
            "answer": [x["answer"] for x in batch],
            "repeat_idx": [x["repeat_idx"] for x in batch],
            "data_idx": [x["data_idx"] for x in batch],
            "global_idx": [x["repeat_idx"] * args.dataset_size + x["data_idx"] for x in batch],
        }

    dataset = RepeatDataset(dataset, args.num_repetitions)
    all_indices = set(list(range(len(dataset))))
    remaining_indices = all_indices - set(skip_indices)
    print(f"Remaining indices to process: {len(remaining_indices)} out of {len(dataset)} total indices.")
    dataset = torch.utils.data.Subset(dataset, sorted(remaining_indices))  # repeat dataset for num_repetitions
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate)
    for batch in loader:
        global_indices = batch["global_idx"]
        print(f"Starting batch of {min(global_indices)} to {max(global_indices)}...")
        batch_t0 = time.time()
        cur_batch_size = len(batch["input_ids"])

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if args.search_type == "bfs":
            beam_size = args.num_blocks
        elif args.search_type.startswith("beam"):
            # e.g. beam4, beam8, etc.
            beam_size = int(args.search_type[4:])
        else:
            raise ValueError(f"Unknown search_type: {args.search_type}")

        bfs_outputs = generate_beam(
            piref_model=piref_model, tokenizer=tokenizer, classifier_model=classifier, prompt_ids=batch["input_ids"],
            max_length=args.max_length, temperature=args.temperature, top_p=args.top_p,
            num_blocks=args.num_blocks, block_size=args.block_size, seed=args.seed,
            beam_size=beam_size,
        )
        generated_ids = bfs_outputs["response_ids"]
        generated_scores = bfs_outputs["response_scores"]

        print("Calculating rewards...")
        is_nested = False
        gt_answers = batch["answer"]
        if isinstance(generated_ids[0][0], list):
            is_nested = True
            flattened_generated_ids = []
            num_generations_per_prompt = []
            flattened_gt_answers = []
            for i in range(len(generated_ids)):
                flattened_generated_ids.extend(generated_ids[i])
                num_generations_per_prompt.append(len(generated_ids[i]))
                flattened_gt_answers.extend([gt_answers[i]] * len(generated_ids[i]))
            generated_ids = flattened_generated_ids
            gt_answers = flattened_gt_answers

        generated_raw_texts = tokenizer.batch_decode(generated_ids)
        generated_solutions = [deepseek_utils.remove_thinking_text(x) for x in generated_raw_texts]
        processed_answers = [accuracy_utils.process_sample(x) for x in generated_solutions]

        num_empty = sum(1 for x in processed_answers if x == "")
        print(f"{num_empty} answers are empty string out of {len(processed_answers)}: {num_empty/len(processed_answers):.2%}.")

        # multiprocessing rewards calculation
        num_workers = 24
        import multiprocessing as mp
        assert len(gt_answers) == len(processed_answers)
        with mp.Pool(num_workers) as pool:
            tasks = zip(gt_answers, processed_answers)
            rewards = list(tqdm(pool.imap(math_verify_wrapper, tasks), total=len(gt_answers)))

        # unflatten
        if is_nested:
            generated_ids = unflatten_variable_n(generated_ids, num_generations_per_prompt)
            generated_raw_texts = unflatten_variable_n(generated_raw_texts, num_generations_per_prompt)
            processed_answers = unflatten_variable_n(processed_answers, num_generations_per_prompt)
            rewards = unflatten_variable_n(rewards, num_generations_per_prompt)

        print("Finished calculating rewards. Now writing to output file...")
        dt = time.time() - batch_t0
        with open(args.output_path, "a") as f:
            for i in tqdm(range(cur_batch_size), desc="writing to file", leave=False):
                outputs = {
                    "repeat_idx": batch['repeat_idx'][i],
                    "data_idx": batch['data_idx'][i],
                    "global_idx": batch["global_idx"][i],
                    "problem": batch["problem"][i],
                    "gt_answer": batch["answer"][i],
                    "generated_ids": generated_ids[i],
                    "generated_scores": generated_scores[i],
                    "processed_answer": processed_answers[i],
                    "reward": rewards[i],
                    "dt": dt / cur_batch_size,  # average time per sample
                }
                f.write(json.dumps(outputs) + "\n")
                f.flush()

        rewards = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards.mean().item()
        se_reward = 0 if len(rewards) == 1 else torch.std(rewards, correction=1) / (len(rewards) ** 0.5)
        print(f"indices={min(global_indices)}-{max(global_indices)} | size={cur_batch_size} | dt={dt/cur_batch_size:.2f}s per elem | reward={mean_reward} ± {se_reward}")

    print("Finished all batches. Now logging to neptune...")
    log_to_server(args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)