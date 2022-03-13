import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import DistilBertPreTrainedModel, DistilBertModel

MASK_TOKEN = -100
CLS_TOKEN = 101
SEP_TOKEN = 102
PAD_TOKEN = 0
ALPHAS = [0.0]

class MLMModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.mlm_probability = 0.15
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        self.vocab_size = None
        self.mask_token = MASK_TOKEN

        self.alphas = ALPHAS
        self.alpha_idx = 0

    def set_mask_token(self, mask_token):
        self.mask_token = mask_token

    def set_alphas(self, alphas):
        self.alphas = alphas
        self.alpha_idx = 0

    def add_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def mlm_mask_inputs(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.vocab_size is None:
            raise AttributeError('Vocab size not specified for MLMModel')

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)

        special_tokens_mask = torch.zeros_like(inputs, device=inputs.device)
        special_tokens_mask[inputs == CLS_TOKEN] = 1 # [CLS], [SEP], and [PAD] can't be masked
        special_tokens_mask[inputs == SEP_TOKEN] = 1
        special_tokens_mask[inputs == PAD_TOKEN] = 1
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decay_alpha=False,
        mask_inputs=False,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        if mask_inputs:
            input_ids, mlm_labels = self.mlm_mask_inputs(input_ids)
        else:
            mlm_labels = input_ids

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict = return_dict
        )

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        # QA logits
        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        # MLM logits
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, max_query_length, dim)
        prediction_logits = F.gelu(prediction_logits)  # (bs, max_query_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, max_query_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, max_query_length, vocab_size)

        # QA Cross-Entropy Loss
        qa_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            qa_start_loss = loss_fct(start_logits, start_positions)
            qa_end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (qa_start_loss + qa_end_loss) / 2

        # MLM Cross-Entropy Loss
        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), mlm_labels.view(-1))

        if self.alpha_idx > len(self.alphas) - 1:
            alpha_cur = self.alphas[-1]
        else:
            alpha_cur = self.alphas[self.alpha_idx]

        if decay_alpha:
            self.alpha_idx += 1

        # total loss
        if qa_loss is None:
            total_loss = alpha_cur * mlm_loss
        else:
            total_loss = qa_loss + alpha_cur *  mlm_loss

        output = (start_logits, end_logits, prediction_logits) + distilbert_output[1:]

        return ((total_loss,) + output) if total_loss is not None else output
