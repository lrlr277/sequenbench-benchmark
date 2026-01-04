import jsonlines
import os
import json
import re
import string
# SRC_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/model_output/ft/llava_run0.jsonl"
SRC_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/model_output/close/gemini_3shot.jsonl"
DST_PATH = f"/mnt/beegfs/xr/lm_multimodal/seq/evaluation/bad_case/gemini_3shot.jsonl"

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

def extract(file):
    inf = read_jsonl(file)
    judge(inf)
    bad_inf = [d for d in inf if not d['correct']]
    for d in bad_inf:
        del d['correct']
    return bad_inf

if __name__ == '__main__':
    dst = extract(SRC_PATH)
    to_jsonl(dst, DST_PATH)
