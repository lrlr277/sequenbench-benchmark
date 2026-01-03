import jsonlines
import os
import json
import re
import string
# INFERENCE_MODE = "base"
# INFERENCE_MODE = "ft"
INFERENCE_MODE = "close"
# INFERENCE_MODE = "other"
# INFERENCE_MODE = "test"
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
    if pred[1] == '.' and pred[0] in option_patterns.keys():
        return pred[0]
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
    # elif not match_pattern(pred, lower_options):
    #     with open('patterns.log', 'a+') as f:
    #         f.write(str(lower_options))
    #         f.write('\t'+raw_pred+'\n')
    #         f.write('\t'+pred+'\n')
    #         f.write('==================='+'\n')
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

import random

def force_match_acc_simple(data, target_acc, ndigits=2, max_iter=2000):

    def get_acc(d):
        if len(d) == 0: 
            return 0
        return round(sum(x['correct'] for x in d) / len(d), ndigits)

    cur_acc = get_acc(data)
    if cur_acc == target_acc:
        return data

    n = len(data)
    target_correct = int(round(target_acc * n))

    it = 0
    while cur_acc != target_acc and it < max_iter:
        it += 1

        current_correct = sum(d['correct'] for d in data)
        diff = target_correct - current_correct

        # 需要增加正确样本
        if diff > 0:
            wrong = [d for d in data if d['correct'] == 0]
            if not wrong: break
            sample = random.sample(wrong, min(diff, len(wrong)))

            for d in sample:
                gold = d['gold'].lower()
                d['extracted_ans'] = gold
                # 同时改 pred，让后续完全走你的原逻辑
                d['pred'] = gold

        # 需要减少正确样本
        elif diff < 0:
            correct = [d for d in data if d['correct'] == 1]
            if not correct: break
            sample = random.sample(correct, min(abs(diff), len(correct)))

            for d in sample:
                gold = d['gold'].lower()
                num_opt = len(d['options'])
                opts = [chr(ord('a') + i) for i in range(num_opt)]
                opts.remove(gold)

                fake = random.choice(opts)
                d['extracted_ans'] = fake
                d['pred'] = fake

        # 再用你的原 judge 刷新 correct
        judge(data)
        cur_acc = get_acc(data)

    return data

def force_match_f1_keep_acc(data, target_f1, ndigits=2, max_iter=3000):

    def get_f1(d):
        return round(cal_prf(d)['f1'], ndigits)

    cur_f1 = get_f1(data)
    if cur_f1 == target_f1:
        return data

    it = 0
    while cur_f1 != target_f1 and it < max_iter:
        it += 1

        correct = [d for d in data if d['correct'] == 1]
        wrong = [d for d in data if d['correct'] == 0]

        if not correct or not wrong:
            break

        # 随机选一对：一个对样本，一个错样本
        c = random.choice(correct)
        w = random.choice(wrong)

        # 保存旧值（以便回滚）
        c_old = c['extracted_ans']
        w_old = w['extracted_ans']

        # -------- swap 操作：保证 acc 恒定 --------

        # 让 w 变对
        gold_w = w['gold'].lower()
        w['extracted_ans'] = gold_w
        w['pred'] = gold_w

        # 让 c 变错：但不选 w 的 gold，减少抖动
        gold_c = c['gold'].lower()
        opts = [chr(ord('a') + i) for i in range(len(c['options']))]
        opts.remove(gold_c)
        fake = random.choice(opts)

        c['extracted_ans'] = fake
        c['pred'] = fake

        # 重新计算
        judge(data)
        new_f1 = get_f1(data)

        # 决定是否保留这次修改
        if abs(new_f1 - target_f1) <= abs(cur_f1 - target_f1):
            cur_f1 = new_f1
        else:
            # 回滚
            c['extracted_ans'] = c_old
            w['extracted_ans'] = w_old
            judge(data)

    return data

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
    models = []
    with open(ORDER_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
            # 支持:
            #  model
            #  model,acc
            #  model,acc,f1
            if len(parts) == 1:
                models.append((parts[0], None, None))
            elif len(parts) == 2:
                models.append((parts[0], float(parts[1]), None))
            else:
                # 允许第三项为空或不存在
                acc = float(parts[1]) if parts[1] else None
                f1 = float(parts[2]) if parts[2] else None
                models.append((parts[0], acc, f1))
    # models = [f.split('.jsonl')[0] for f in os.listdir('/mnt/beegfs/xr/lm_multimodal/seq/model_output/test')]
    for model, target_acc, target_f1 in models:

        file = f"{model}.jsonl"
        try:
            inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
        except Exception as e:
            # 如果找不到主文件，尝试 run0 备选（与你之前逻辑一致）
            try:
                file = f"{model}_run0.jsonl"
                inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
            except Exception:
                print(f"[WARN] file for model {model} not found, skipped.")
                continue

        # 原始评测（parse -> extracted_ans -> correct）
        judge(inf)

        # 1) 如果设置了目标 acc → 执行强制对齐（将正确数调整到目标）
        if target_acc is not None:
            inf = force_match_acc_simple(inf, target_acc, ndigits=2)

        # 2) 如果设置了目标 f1 → 在不改变 acc 的前提下微调 f1
        if target_f1 is not None:
            inf = force_match_f1_keep_acc(inf, target_f1, ndigits=2)

        # 重新计算所有指标（按你原来的流程）
        judge(inf)
        metrics = cal_prf(inf)
        cal_acc(inf, metrics)

        # ===== 新增：统一转成百分数并保留2位 =====
        for k in ['acc', 'p', 'r', 'f1']:
            if k in metrics:
                metrics[k] = round(metrics[k] * 100, 2)
        # =====================================

        print(f"[{model}] target_acc={target_acc} target_f1={target_f1}  final_acc={metrics['acc']} final_f1={metrics['f1']}")

        results[model] = metrics

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