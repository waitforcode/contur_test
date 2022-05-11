import os.path

import torch
from sklearn.metrics import classification_report
from BertClf import BertClf
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any


def predict_metrics(
        model: BertClf,
        iterator: torch.utils.data.DataLoader,
):
    """
    Применение модели на тестовой выборке и рассчет метрик

    :param model: модель для тюнинга
    :param iterator: loader
    """

    true = []
    pred = []

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc="Computing final metrics..."):
            predictions = model(texts.to(model.device)).squeeze()
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()
            true.extend(y_true)
            pred.extend(preds)

    true = [model.mapper[x] for x in true]
    pred = [model.mapper[x] for x in pred]

    print(classification_report(true, pred, zero_division=0))


def train(
        model: BertClf,
        iterator: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        criterion: torch.nn
) -> float:
    """
    Одна эпоха обучения модели

    :param model: модель для тюнинга
    :param iterator: loader
    :param optimizer:
    :param criterion:
    :return: среднее значение метрики
    """

    epoch_loss = []
    epoch_f1 = []

    model.train()

    for texts, ys in tqdm(iterator, total=len(iterator), desc='Training loop'):
        optimizer.zero_grad()
        predictions = model(texts.to(model.device))
        loss = criterion(predictions, ys.to(model.device))

        loss.backward()
        optimizer.step()
        preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
        y_true = ys.tolist()

        epoch_loss.append(loss.item())
        epoch_f1.append(f1_score(y_true, preds))

    return np.mean(epoch_f1)


def evaluate(
        model: BertClf,
        iterator: torch.utils.data.DataLoader,
        criterion: torch.nn
) -> float:
    """
    :param model: модель
    :param iterator: torch.utils.data.DataLoader
    :param criterion: instance of torch-like loses
    :return: среднее значение метрики
    """
    epoch_loss = []
    epoch_f1 = []

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc='Evaluating loop'):
            predictions = model(texts.to(model.device))
            loss = criterion(predictions, ys.to(model.device))
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()

            epoch_loss.append(loss.item())
            epoch_f1.append(f1_score(y_true, preds))

    return np.mean(epoch_f1)


def train_evaluate(
        model: BertClf,
        training_generator: torch.utils.data.DataLoader,
        valid_generator: torch.utils.data.DataLoader,
        criterion: torch.optim,
        optimizer: torch.nn,
        num_epochs: int,
        out_path: Path
):
    """
    Процесс обучения модели
    :param out_path:
    :param model: класс (архитектура) модели
    :param training_generator: train
    :param valid_generator: validation
    :param criterion: loss from torch losses
    :param optimizer: optimizer from torch optimizers
    :param num_epochs: число эпох,
    :return: натренированная модель
    """
    max_eval_f1 = 0
    for i in range(num_epochs):

        print(f"==== Epoch {i+1} out of {num_epochs} ====")
        train_f1 = train(
            model=model,
            iterator=training_generator,
            optimizer=optimizer,
            criterion=criterion
        )

        eval_f1 = evaluate(
            model=model,
            iterator=valid_generator,
            criterion=criterion
        )

        if eval_f1 > max_eval_f1:
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            torch.save(model.state_dict(), f'{out_path}/model_epoch_{i+1}_f1_{eval_f1}.pth')
            max_eval_f1 = eval_f1

        print(f'Train F1: {train_f1}\nEval F1: {"{:10.4f}".format(eval_f1)}')
        print()

    print()
    predict_metrics(
        model=model,
        iterator=valid_generator
    )
    return model
