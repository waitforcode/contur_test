import torch
from torch.utils.data import Dataset
from typing import List, Union, Tuple
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class NewsDataset(Dataset):
    """
    Токенизация и преобразование текста
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            maxlen: int,
            texts: List[str],
            targets: Union[List[str], List[int]]
    ):
        self.tokenizer = tokenizer
        self.texts = [torch.LongTensor(self.tokenizer.encode(t, truncation=True, max_length=maxlen)) for t in texts]
        self.texts = torch.nn.utils.rnn.pad_sequence(self.texts, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)
        self.length = len(texts)
        self.targets = torch.LongTensor(targets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        ids = self.texts[item]
        y = self.targets[item]
        return ids, y
