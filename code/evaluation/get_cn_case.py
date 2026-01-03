import jsonlines

name = "llava"
src_file = f"bad_case/{name}.jsonl"
ref_file = "/mnt/beegfs/xr/lm_multimodal/seq/data/all_cn.jsonl"
dst_file = f"bad_case/{name}_cn.jsonl"

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
ref_dict = {d['image']: d for d in ref}
dst = []
for d in src:
    assert ref_dict[d['image']]['answer'] == d['gold']
    d['question'] = ref_dict[d['image']]['question']
    d['id'] = ref_dict[d['image']]['id']
    d['options'] = ref_dict[d['image']]['options']
    dst.append(d)
to_jsonl(dst, dst_file)