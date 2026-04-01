import jsonlines
import random

all_file = "all.jsonl"
train_file = "split_73/train.jsonl"
val_file = "split_73/val.jsonl"
test_file = "split_73/test.jsonl"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    print(len(data), data[0].keys())
    return data

def to_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)

all_data = read_jsonl(all_file)

random.shuffle(all_data)

n = len(all_data)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_data = all_data[:train_end]
val_data = all_data[train_end:val_end]
test_data = all_data[val_end:]

to_jsonl(train_data, train_file)
to_jsonl(val_data, val_file)
to_jsonl(test_data, test_file)

print(f"Total: {n}, Train: {len(train_data)}, Dev: {len(val_data)}, Test: {len(test_data)}")
