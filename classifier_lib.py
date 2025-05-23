from transformers import Qwen2PreTrainedModel, Qwen2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CustomSequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    # Prob of reward being 1
    success_probs: Optional[torch.FloatTensor] = None


class Qwen2ForClassifier(Qwen2PreTrainedModel):
    def __init__(self, config, use_bias=False):
        super().__init__(config)
        num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=use_bias),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_labels, bias=use_bias),
        )
        self.p_dropout = config.attention_dropout
        self.score_dropout = nn.Dropout(self.p_dropout)
        self.inference_impl = "naive"
        self.train_bt_model = False
        self.num_labels = num_labels

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
        continuation_ids: Optional[torch.LongTensor] = None,
        continuation_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CustomSequenceClassifierOutputWithPast]:
        """
        During training:
        - labels should not be None and have shape: [bs, 1]
        - input_ids: [bs, seqlen]
        - loss_mask [bs, seqlen]

        During inference:
        labels, loss_mask should be None
        continuation_ids is [bs, N, c_len].
        If input_ids is [bs, seqlen], this is prefill stage.
        Otherwise, input_ids is also [bs, c_len] which contains the chosen continuation from last step. And we update the kv_cache.
        Here, attention_mask should be [bs, q_len] where q_len is seqlen + len of continuations so far.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict, "Only return_dict=True is supported."
        is_training = labels is not None
        is_single_eval = continuation_ids is None
        if not is_training: assert not self.training, "Model should not be in training mode during inference."

        if is_training:
            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim]
            logits = self.score(self.score_dropout(hidden_states)).float()  # [bs, seqlen, num_labels]
            bs, seqlen, _ = logits.shape
            if self.train_bt_model:
                assert self.num_labels == 1, f"BT model should have 1 label. Got {self.num_labels}."
                assert bs % 2 == 0, f"Batch size should be even for BT model. Got {bs}."
                logits = logits[:, -1, 0]  # [bs, seqlen, 1] -> [bs]
                # bt loss
                assert torch.all(labels[::2] == 1), f"Labels should be 1 for chosen logits. Got {labels[::2]}."
                assert torch.all(labels[1::2] == 0), f"Labels should be 0 for rejected logits. Got {labels[1::2]}."
                chosen_logits = logits[::2]  # [bs//2]
                reject_logits = logits[1::2]  # [bs//2]
                elemwise_loss = -F.logsigmoid(chosen_logits - reject_logits)  # [bs//2]
                loss = elemwise_loss.mean()
            else:
                if self.num_labels == 1:
                    # BCE Loss
                    labels_expanded = labels.unsqueeze(-1).expand_as(logits)
                    elemwise_loss = F.binary_cross_entropy_with_logits(logits, labels_expanded, reduction="none")  # [bs, seqlen]
                else:
                    # CrossEntropyLoss
                    labels_expanded = labels.long().unsqueeze(-1).expand((bs, seqlen))  # [bs, seqlen]
                    elemwise_loss = F.cross_entropy(
                        logits.transpose(1, 2),  # [bs, seqlen, num_labels] -> [bs, num_labels, seqlen]
                        labels_expanded,  # [bs, seqlen]
                        reduction="none",
                    )
                # avg over seqlen and bs. do so in a way that prevents nans from division by zero
                mask_sum = loss_mask.sum(1).float()
                safe_denom = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
                loss = torch.where(mask_sum > 0, (elemwise_loss * loss_mask).sum(1) / safe_denom, mask_sum)  # [bs]
                loss = loss.mean()

            if torch.isnan(loss).any():
                breakpoint()
            return CustomSequenceClassifierOutputWithPast(loss=loss, logits=logits)

        elif is_single_eval:
            # single eval is also useful for updating kv_cache
            assert continuation_ids is None
            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim]
            logits = self.score(hidden_states).float()  # [bs, seqlen, num_labels]
            if logits.shape[-1] > 1:
                # assume 1 is the index/label for success
                success_probs = F.softmax(logits, dim=-1)[:, :, 1]  # [bs, seqlen]
            else:
                assert logits.shape[-1] == 1, f"Expected logits to have 1 output, got {logits.shape}."
                # TODO: change back to sigmoid!
                # success_probs = logits.squeeze(-1)  # [bs, seqlen]
                success_probs = logits.squeeze(-1).sigmoid()  # [bs, seqlen]

            return CustomSequenceClassifierOutputWithPast(
                logits=logits, success_probs=success_probs, past_key_values=transformer_outputs.past_key_values)


