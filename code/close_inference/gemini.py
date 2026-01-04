from openai import OpenAI
import os
import base64
from util import TEST_DATA_PATH, RESULT_DIR, construct_few_shot, RERUN_DATA_DICT
import os
import json
from tqdm import tqdm

os.makedirs(RESULT_DIR, exist_ok=True)
num_shot = 1
mode = f"gemini_{num_shot}shot"
result_file = f"gemini_{num_shot}shot.jsonl"

# 跑rerun_circular
# TEST_DATA_PATH = RERUN_DATA_DICT[mode]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="xxxx",
)

def chat(entry, num_shot=0):
    messages = [{"role": "user", "content": construct_few_shot(entry, num_shot, 'gemini')}]
    completion = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=messages,
        temperature=0,
        seed=42
    )
    return completion.choices[0].message.content

with open(TEST_DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'a+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    cnt = 0
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        cnt += 1
        if cnt <= 143: continue
        entry = json.loads(line)
        pred = chat(entry, num_shot)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {"image": entry['image'], "question": entry['question'], "options": entry['options'], "pred": pred, "gold": entry['answer']}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        # break