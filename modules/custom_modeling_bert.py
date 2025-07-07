
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.utils import logging
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BertForSentipolcClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.pos_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.neg_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        label_pos: Optional[torch.Tensor] = None,
        label_neg: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)


        pos_logits = self.pos_classifier(pooled_output)
        neg_logits = self.neg_classifier(pooled_output)

        loss = None
        loss_fct = CrossEntropyLoss()
        pos_loss = loss_fct(pos_logits.view(-1, self.num_labels), label_pos.view(-1))
        neg_loss = loss_fct(neg_logits.view(-1, self.num_labels), label_neg.view(-1))
            
        loss = (pos_loss + neg_loss) / 2

        return SequenceClassifierOutput(
            loss=loss,
            logits={'pos': pos_logits, 'neg':neg_logits},
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )