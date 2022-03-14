import math

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, GELU

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import DistilBertForMaskedLM

class DistilBertForMaskedLMQA(DistilBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()

        # add them after FewShotDataset is initialized with the right label.
        self.label_word_list = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_pos=None,
        start_positions=None,
        end_positions=None,
    ):
        gelu = GELU()

        assert mask_pos is not None
        assert mask_pos.size(dim=1) == 2
        assert mask_pos.size(dim=0) == input_ids.size(dim=0)
        mask_start = mask_pos[:,0] # (bs,)
        mask_end = mask_pos[:,1] # (bs,)

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        
        # Return logits for only the two masked locations
        start_logits = prediction_logits[torch.arange(prediction_logits.size(0)), mask_start, :] # (bs, vocab_size), natually squeezed.
        end_logits = prediction_logits[torch.arange(prediction_logits.size(0)), mask_end, :] # (bs, vocab_size)

        # Return logits for each label
        label_start_logits = []
        label_end_logits = []
        for label_id in range(len(self.label_word_list)):
            label_start_logits.append(start_logits[:, self.label_word_list[label_id]].unsqueeze(-1)) # (bs, 1), the last 1 is preserved
            label_end_logits.append(end_logits[:, self.label_word_list[label_id]].unsqueeze(-1)) # (bs, 1), the last 1 is preserved

        start_logits = torch.cat(label_start_logits, -1) # (bs, label_size)
        end_logits = torch.cat(label_end_logits, -1) # (bs, label_size)
        # print("start_logits: %s", start_logits.size())
        # print("end_logits: %s", end_logits.size())

        mlm_loss = None
        if start_positions is not None:
            loss_fct = CrossEntropyLoss()
            # print("start_positions.view(-1): ", start_positions.view(-1))
            # print("end_positions.view(-1): ", end_positions.view(-1))
            start_loss = loss_fct(start_logits.view(-1, start_logits.size(-1)), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, end_logits.size(-1)), end_positions.view(-1))
            mlm_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=mlm_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )
