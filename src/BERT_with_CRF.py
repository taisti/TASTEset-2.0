import torch
from typing import Optional
from transformers import BertForTokenClassification
from transformers.utils import ModelOutput
from torchcrf import CRF


class BertCRFOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    y_preds: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None


class BERTCRF(BertForTokenClassification):
    """
    BERT with CRF layer on top. This class directly follows implementation used
    in "BERTimbau: Pretrained BERT Models for Brazilian Portuguese", which was
    made available under the MIT license and can be found under the following
    link:  https://github.com/neuralmind-ai/portuguese-bert
    """

    def __init__(self, config):
        """
        :param config: BertConfig
        """
        super().__init__(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, prediction_mask=None, ):

        outputs = {}

        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]

        output = self.dropout(output)
        logits = self.classifier(output)

        mask = prediction_mask
        batch_size = logits.shape[0]

        outputs['logits'] = logits
        if labels is not None:
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                seq_mask = [i for i, mask in enumerate(seq_mask) if mask]
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels, reduction='token_mean')
            loss /= batch_size
            outputs['loss'] = loss

        else:
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_mask = [i for i, mask in enumerate(seq_mask) if mask]
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                output_tags.append(tags[0])

            output_tags = [tags + [-100] * (128 - len(tags)) for tags in
                           output_tags]
            outputs['predictions'] = torch.tensor(output_tags)

        return BertCRFOutput(**outputs)
