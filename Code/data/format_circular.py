import jsonlines

DATA_PATH = "split_73/test.jsonl"
DST_PATH = "split_73/test_circular.jsonl"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    # print(len(data), data[0].keys())
    return data

def to_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)

if __name__ == '__main__':
    data = read_jsonl(DATA_PATH)
    new_data = []
    new_sequential = []
    for d in data:
        tmp = []
        num_options = len(d['options'])
        options_text = [o[1:] for o in d['options']]
        for i in range(num_options-1):
            options = [f"{chr(ord('A')+idx)}{options_text[(i+1+idx)%num_options]}" for idx in range(num_options)]
            answer = chr(ord('A')+(ord(d['answer'])-ord('A')-i-1)%num_options)
            tmp.append({**d, "options": options, "answer": answer})
            new_sequential.append({**d, "options": options, "answer": answer})
        assert len(tmp) == num_options-1
        new_data.append(tmp)
    # to_jsonl(new_data, DST_PATH1)
    to_jsonl(new_sequential, DST_PATH)