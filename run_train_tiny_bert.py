import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from train_utils import train_evaluate
from BertClf import BertClf
from pathlib import Path
from processing_utils import get_train_val_and_mapper, get_data_loaders, read_csv


data_path = 'data/my_train.tsv'
model_name = "cointegrated/rubert-tiny"


def train(path_to_data):
    """
    path_to_data: путь к тренировочным данным
    """

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model_bert = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)

    df = read_csv(Path(path_to_data))
    id2label, train_texts, valid_texts, train_targets, valid_targets = get_train_val_and_mapper('title', 'is_fake', df)

    model = BertClf(model=model_bert, tokenizer=tokenizer, id_to_label=id2label, dropout=0.2, device=device)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.NLLLoss()

    training_generator, valid_generator = get_data_loaders(
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_targets=train_targets,
        valid_texts=valid_texts,
        valid_targets=valid_targets,
        batch_size=64,
        maxlen=512
    )

    model = train_evaluate(
        model=model,
        training_generator=training_generator,
        valid_generator=valid_generator,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        out_path=Path('./model/')
    )

    with open(os.path.join('./model/', 'label_mapper.json'), mode='w', encoding='utf-8') as f:
        json.dump({int(k): str(v) for k, v in model.mapper.items()}, f, ensure_ascii=False)


def main():
    train(data_path)


if __name__ == "__main__":
    main()
