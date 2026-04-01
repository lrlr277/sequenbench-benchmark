import jsonlines
import json
import pandas as pd

DATA_PATH = {
    "all": "/mnt/beegfs/xr/lm_multimodal/seq/data/all.jsonl",
    "train": "/mnt/beegfs/xr/lm_multimodal/seq/data/split_73/train.jsonl",
    "val": "/mnt/beegfs/xr/lm_multimodal/seq/data/split_73/val.jsonl",
    "test": "/mnt/beegfs/xr/lm_multimodal/seq/data/split_73/test.jsonl"
}
CSV_PATH = f"output/stat_dataset.csv"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    # print(len(data), data[0].keys())
    return data

def to_csv(stats, file_path):
    rows = []
    splits = ['train', 'val', 'test', 'all', 'ratio']
    for key in stats['train']:
        row = [key]
        for split in splits:
            row.append(stats[split][key])
        rows.append(row)
    df = pd.DataFrame(rows, columns=[
    "", "Train", "Dev", "Test", "Total", "Ratio"
    ])
    df.to_csv(file_path, index=False, encoding="utf-8")

def stat_num(data):
    stat = {}
    # 总体数据
    stat['all'] = len(data)
    # 选项
    stat['a'] = len([d for d in data if d['answer'].lower()=='a'])
    stat['b'] = len([d for d in data if d['answer'].lower()=='b'])
    stat['c'] = len([d for d in data if d['answer'].lower()=='c'])
    stat['d'] = len([d for d in data if d['answer'].lower()=='d'])
    return stat

def stat_ratio(stats):
    stats['ratio'] = {}
    for key in stats['all']:
        stats['ratio'][key] = round(stats['all'][key]/stats['all']['all'] * 100, 2)

stats = {}
for key in DATA_PATH:
    split = read_jsonl(DATA_PATH[key])
    stats[key] = stat_num(split)
stat_ratio(stats)
print(stats)
to_csv(stats, CSV_PATH)