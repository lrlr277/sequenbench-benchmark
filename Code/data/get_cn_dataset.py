import jsonlines

src_file = "split_73/test_200.jsonl"
ref_file = "all_cn.jsonl"
dst_file = "split_73/test_200_cn.jsonl"

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

src = read_jsonl(src_file)
ref = read_jsonl(ref_file)
ref_dict = {d['id']: d for d in ref}
dst = []
for d in src:
    assert ref_dict[d['id']]['answer'] == d['answer']
    dst.append(ref_dict[d['id']])
to_jsonl(dst, dst_file)