import jsonlines
import os
import json
import re
import string
# INFERENCE_MODE = "base"
# INFERENCE_MODE = "ft"
INFERENCE_MODE = "close"
INFERENCE_DIR = f"/mnt/beegfs/xr/lm_multimodal/seq/model_output/circular_{INFERENCE_MODE}"
REF_DIR = f"/mnt/beegfs/xr/lm_multimodal/seq/model_output/{INFERENCE_MODE}"
ORDER_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/model_order/model_order_circular_{INFERENCE_MODE}.txt"
DST_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/output/circular_{INFERENCE_MODE}.json"
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

def is_answer(s):
    patterns1 = ['a', 'b', 'c', 'd']
    patterns2 = ['a.', 'b.', 'c.', 'd.']
    for pattern in patterns1:
        if s == pattern: return True
    if len(s) < 15:
        for pattern in patterns2:
            # if s == 'a. left':
            # print(s.startswith(pattern), pattern)
            if s.startswith(pattern): return True
    return False

def match_pattern(pred, options):
    labels = [o.split('.')[0].strip() for o in options]
    texts = [o.split('.')[-1].strip() for o in options]
    option_patterns = {chr(ord('a')+i): [options[i], texts[i], ','.join(texts[i]), '<'.join(texts[i]), '>'.join(texts[i])] for i in range(4)}
    for key in option_patterns:
        for op in option_patterns[key]:
            if op in pred:
                return key
    if pred[1] == '.' and pred[0] in option_patterns.keys():
        return pred[0]
    return None

def parse_pred(d):
    pattern = re.compile(f'\s?([{re.escape(string.punctuation)}])\s?')

    options = d['options']
    lower_options = [pattern.sub(r'\1', o.lower()) for o in options]
    raw_pred = d['pred'].lower()
    pred = pattern.sub(r'\1', raw_pred)

    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is:",
        "answer is",
        "answer:",
        "correct sequence is:",
        "correct sequence is",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in pred:
            pred = pred.split(answer_pattern)[-1].strip()
    re_answer_patterns = [r'.* is:?[\s\n]*((a|b|c|d)\..*)', r'.* option:?[\s\n]*((a|b|c|d)\..*)']
    for re_answer_pattern in re_answer_patterns:
        match = re.match(re_answer_pattern, pred, re.DOTALL)
        if match:
            pred = match.group(1)

    if '\n' in pred:
        sent_start = pred.split('\n')[0].strip()
        sent_end = pred.split('\n')[-1].strip()
        pred = sent_start + '\n' + sent_end

    if is_answer(pred):
        return pred
    # elif not match_pattern(pred, lower_options):
    #     with open('patterns.log', 'a+') as f:
    #         f.write(str(lower_options))
    #         f.write('\t'+raw_pred+'\n')
    #         f.write('\t'+pred+'\n')
    #         f.write('==================='+'\n')
    return match_pattern(pred, lower_options)


def judge(data, ref):
    for d in data:
        pred = parse_pred(d)
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

    for d in ref:
        pred = parse_pred(d)
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

        # if not d['correct'] and extracted_ans != d['pred'][0].lower():
        #     print(gold, extracted_ans, d['pred'])

def cal_acc(inf, ref):
    inf_idx = 0
    num_correct = 0
    for d in ref:
        circular_list = [d]
        while inf_idx < len(inf):
            if inf[inf_idx]['image'] == d['image']:
                circular_list.append(inf[inf_idx])
                inf_idx += 1
            else:
                break
        assert len(circular_list) == len(d['options'])
        if len([x for x in circular_list if x['correct']]) == len(circular_list):
            num_correct += 1
    return {'acc': round(num_correct / len(ref) * 100, 2)}

def eval_one(model):
    files = sorted(os.listdir(INFERENCE_DIR))
    ref_files = sorted(os.listdir(REF_DIR))
    try:
        file = f"{model}_run0.jsonl"
        assert file in files
    except:
        file = f"{model}.jsonl"
        assert file in files
    inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
    try:
        ref_file = f"{model}_run0.jsonl"
        assert ref_file in ref_files
    except:
        ref_file = f"{model}.jsonl"
        assert ref_file in ref_files
    ref = read_jsonl(os.path.join(REF_DIR, ref_file))
    judge(inf, ref)
    metrics = cal_acc(inf, ref)
    return metrics

if __name__ == '__main__':
    results = {}
    with open(ORDER_PATH, 'r') as f:
        models = [item.strip() for item in f.readlines() if item]
    # models = [f.split('.jsonl')[0] for f in os.listdir('/mnt/beegfs/xr/lm_multimodal/seq/model_output/test')]
    for model in models:
        results[model] = eval_one(model)
    with open(DST_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # files = sorted(os.listdir(INFERENCE_DIR))
    # for file in files:
    #     # if file == "intern_8b.jsonl": continue
    #     print("================")
    #     print(file)
    #     inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
    #     judge(inf)
    #     metrics = cal_prf(inf)
    #     cal_acc(inf, metrics)
    #     print(metrics)
    #     # break