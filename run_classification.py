import torch
from processing_utils import get_data, get_data_loader, read_csv
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from train_utils import predict_metrics
from BertClf import BertClf
from sklearn.metrics import f1_score
import json


data_path = 'data/my_test.tsv'
model_path = 'model/model_epoch_20_f1_0.8689704902001734.pth'
mapper_path = 'model/label_mapper.json'
model_name = "cointegrated/rubert-tiny"
max_len = 512


def make_classification(path_to_data, mode='class', res_name='test_bert'):
    """
    :param res_name:
    :param path_to_data:
    :param mode:
    :return:
    """
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model_bert = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)

    with open(mapper_path) as file:
        id2label = {int(k): v for k, v in json.loads(file.read()).items()}
    model = BertClf(model_bert, tokenizer, id_to_label=id2label, dropout=0.2, device=device)
    model.load_state_dict(torch.load(model_path))

    df = read_csv(Path(path_to_data))
    texts = df['title'].tolist()
    predict = []
    for text in texts:
        predict.append(model.predict(text))
    if mode == 'test':
        targets = df['is_fake'].apply(lambda x: str(x)).tolist()
        print(f1_score(targets, predict, pos_label='1'))

    df['bert_is_fake'] = predict
    df.to_csv(f'data/{res_name}.tsv', sep='\t')


def main():
    make_classification('data/my_test.tsv', mode='test', res_name='valid_bert')
    make_classification('data/test.tsv', res_name='result_bert')


if __name__ == "__main__":
    main()
