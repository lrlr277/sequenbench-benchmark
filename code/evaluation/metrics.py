import jsonlines
import os
import json
import re
import string
# INFERENCE_MODE = "base"
# INFERENCE_MODE = "ft"
# INFERENCE_MODE = "close"
# INFERENCE_MODE = "other"
# INFERENCE_MODE = "test"
# INFERENCE_MODE = "cot_base"
INFERENCE_MODE = "cot_ft"
INFERENCE_DIR = f"/mnt/beegfs/xr/lm_multimodal/seq/model_output/{INFERENCE_MODE}"
ORDER_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/model_order/model_order_{INFERENCE_MODE}.txt"
# ORDER_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/model_order/model_order_size_{INFERENCE_MODE}.txt"
DST_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/output/{INFERENCE_MODE}.json"

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
    try:
        if pred[1] == '.' and pred[0] in option_patterns.keys():
            return pred[0]
    except:
        return None
    return None

def parse_pred(d):
    pattern = re.compile(f'\s?([{re.escape(string.punctuation)}])\s?')

    options = d['options']
    lower_options = [pattern.sub(r'\1', o.lower()) for o in options]
    raw_pred = d['pred'].lower().split('\n\n')[-1].strip()
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
    elif not match_pattern(pred, lower_options):
        with open('patterns.log', 'a+') as f:
            f.write(str(lower_options))
            f.write('\t'+raw_pred+'\n')
            f.write('\t'+pred+'\n')
            f.write('==================='+'\n')
    return match_pattern(pred, lower_options)


def judge(data):
    for d in data:
        pred = parse_pred(d)
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

        # if not d['correct'] and extracted_ans != d['pred'][0].lower():
        #     print(gold, extracted_ans, d['pred'])

def init_cnt(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'tp': 0, 'fp': 0, 'fn': 0}
    return d

def cal_prf(inf):
    counter = init_cnt(4)
    for d in inf:
        if not d['extracted_ans']:
            counter[d['gold'].lower()]['fn'] += 1
        else:
            if d['correct']:
                counter[d['extracted_ans']]['tp'] += 1
            else:
                counter[d['extracted_ans']]['fp'] += 1
                counter[d['gold'].lower()]['fn'] += 1
    def compute_macro_prf(cnt, num_options):
        ps, rs, f1s = [], [], []
        for c in cnt.values():
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        return sum(ps)/num_options, sum(rs)/num_options, sum(f1s)/num_options

    p, r, f1 = compute_macro_prf(counter, 4)
    return {
        'p': p,
        'r': r,
        'f1': f1,
    }

def init_cnt_acc(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'correct': 0, 'total': 0}
    return d

def cal_acc(inf, metrics):
    num_correct = len([d for d in inf if d['correct']])
    metrics['acc'] = num_correct / len(inf)
    counter = {str(i+1): init_cnt_acc(4) for i in range(9)}
    for d in inf:
        entry_type = d['image'][0]
        if d['correct']:
            counter[entry_type][d['gold'].lower()]['correct'] += 1
        counter[entry_type][d['gold'].lower()]['total'] += 1
    for entry_type in counter:
        counter[entry_type]['acc'] = sum([m['correct'] for m in counter[entry_type].values()]) / sum([m['total'] for m in counter[entry_type].values()])
        for i in range(4):
            counter[entry_type][chr(ord('a') + i)]['acc'] = counter[entry_type][chr(ord('a') + i)]['correct'] / counter[entry_type][chr(ord('a') + i)]['total'] if counter[entry_type][chr(ord('a') + i)]['total'] else 0
            # counter[entry_type][chr(ord('a') + i)] = counter[entry_type][chr(ord('a') + i)]['correct'] / counter[entry_type][chr(ord('a') + i)]['total']
    metrics['acc_per_type'] = counter

def merge_metrics(dicts, ndigits=2):
    def recursive_avg(ds):
        result = {}
        for k in ds[0].keys():
            values = [d[k] for d in ds]
            if isinstance(values[0], dict):
                result[k] = recursive_avg(values)
            else:
                avg = sum(values) / len(values)
                result[k] = round(avg*100, ndigits) if k in ['acc', 'p', 'r', 'f1'] else int(avg)
        return result
    return recursive_avg(dicts)

def eval_multirun(model, num_run=3):
    files = sorted(os.listdir(INFERENCE_DIR))
    model_metrics = []
    with open('patterns.log', 'a+') as f:
        f.write(model+'\n')
        f.write('==================='+'\n')
    if num_run == 1:
        print(f"=========={model}==========")
        try:
            file = f"{model}.jsonl"
            assert file in files
        except:
            file = f"{model}_run0.jsonl"
        inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
        judge(inf)
        metrics = cal_prf(inf)
        cal_acc(inf, metrics)
        model_metrics.append(metrics)
        return merge_metrics(model_metrics)
    else:
        print(f"=========={model}==========")
        for idx in range(num_run):
            file = f"{model}_run{idx}.jsonl"
            assert file in files
            inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
            judge(inf)
            metrics = cal_prf(inf)
            cal_acc(inf, metrics)
            model_metrics.append(metrics)
        return merge_metrics(model_metrics)

if __name__ == '__main__':
    results = {}
    with open(ORDER_PATH, 'r') as f:
        models = [item.strip() for item in f.readlines() if item]
    # models = [f.split('.jsonl')[0] for f in os.listdir('/mnt/beegfs/xr/lm_multimodal/seq/model_output/test')]
    for model in models:
        try:
            results[model] = eval_multirun(model)
        except:
            results[model] = eval_multirun(model, 1)
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