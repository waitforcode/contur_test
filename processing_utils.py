import pandas as pd
from sklearn.model_selection import train_test_split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
from dataset import NewsDataset
from pathlib import Path
import os
from typing import List, Dict, Tuple


def get_train_val_and_mapper(text_col: str, target_col: str, df: pd.DataFrame, test_size: float = None
                             ) -> Tuple[Dict[int, str], List[str], List[str], List[int], List[int]]:
    """
    Разбиение данных на тренировочный и валидационный набор, создание маппинга лейблов в id

    :param test_size:
    :param df:
    :param text_col:
    :param target_col:
    :return: mapper и тренировочный и валидационный массивы
    """
    id2label = {i: l for i, l in enumerate(df[target_col].unique())}
    label2id = {l: i for i, l in id2label.items()}
    texts = df[text_col].to_list()
    targets = df[target_col].map(label2id).to_list()
    train_texts, valid_texts, train_targets, valid_targets = train_test_split(texts, targets, test_size=test_size,
                                                                              random_state=42)

    return id2label, train_texts, valid_texts, train_targets, valid_targets


def get_data(text_col: str, target_col: str, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Получение массива данных и таргетов (и преобразование их в int) из pd.DataFrame
    :param df:
    :param text_col:
    :param target_col:
    :return: mapper и тренировочный и валидационный массивы
    """
    id2label = {i: l for i, l in enumerate(df[target_col].unique())}
    label2id = {l: i for i, l in id2label.items()}
    texts = df[text_col].to_list()
    targets = df[target_col].map(label2id).to_list()

    return texts, targets


def get_data_loaders(
        tokenizer: PreTrainedTokenizerBase,
        train_texts: List[str],
        train_targets: List[int],
        valid_texts: List[str],
        valid_targets: List[int],
        maxlen: int,
        batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Тренировочный и валидационный лоадеры
    :param tokenizer:
    :param train_texts:
    :param train_targets:
    :param valid_texts:
    :param valid_targets:
    :param maxlen:
    :param batch_size:
    :return:
    """
    return get_data_loader(tokenizer, train_texts, train_targets, maxlen, batch_size), \
           get_data_loader(tokenizer, valid_texts, valid_targets, maxlen, batch_size)


def read_csv(path: Path) -> pd.DataFrame:
    """
    Чтение *.csv
    :param path:
    :return:
    """
    _, ext = os.path.splitext(path)
    sep = ','
    if ext.lower() == '.tsv':
        sep = '\t'
    return pd.read_csv(path, sep=sep, encoding='utf-8')


def get_data_loader(
        tokenizer: PreTrainedTokenizerBase,
        texts: List[str],
        targets: List[int],
        maxlen: int,
        batch_size: int
) -> torch.utils.data.DataLoader:
    """

    :param tokenizer:
    :param texts:
    :param targets:
    :param maxlen:
    :param batch_size:
    :return:
    """

    dataset = NewsDataset(
        tokenizer=tokenizer,
        texts=texts,
        targets=targets,
        maxlen=maxlen
    )

    generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return generator
