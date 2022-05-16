import torch
import torch.nn as nn
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


class BertClf(nn.Module):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, id_to_label: Dict[int, str],
                 dropout: float, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LogSoftmax(1)
        self.mapper = id_to_label
        self.device = device

        out_tiny = 312
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(out_tiny, 512)
        self.fc2 = nn.Linear(512, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()
        hidden = self.model(texts, attention_mask=mask)[0]
        cls = hidden[:, 0]

        x = self.fc1(cls)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        outputs = self.act(x)

        return outputs

    def predict(self, text: str):
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = self(inputs)
        pred = outputs.argmax(1).item()
        pred_text = self.mapper[pred]
        return pred_text
