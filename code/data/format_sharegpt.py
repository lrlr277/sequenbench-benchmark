import jsonlines
from util import ROLE_PROMPT

split = 'train'
data_file = f"split_712/{split}.jsonl"
dst_file = f"split_712/{split}_sharegpt_instructblip.jsonl"

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

def transform(data, mode='sharegpt'):
    new_data = []
    for d in data:
        if mode == 'sharegpt':
            # new_d = {"messages": [{"role": "user", "content": f"{ROLE_PROMPT}Input: Image: <image>\nQuestion: {d['question']}, Options: {'; '.join(d['options'])}.\nOutput:"}, {"role": "assistant", "content": d['answer']}], "images": [f"/mnt/beegfs/xr/lm_multimodal/seq/data/images/{d['image']}"]}
            new_d = {"messages": [{"role": "user", "content": f"{ROLE_PROMPT}Input: Question: {d['question']}, Options: {'; '.join(d['options'])}.\nOutput:"}, {"role": "assistant", "content": d['answer']}], "images": [f"/mnt/beegfs/xr/lm_multimodal/seq/data/images/{d['image']}"]}
        else:
            new_d = {}
        new_data.append(new_d)
    return new_data

to_jsonl(transform(read_jsonl(data_file)), dst_file)