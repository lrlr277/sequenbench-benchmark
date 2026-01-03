from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, CIRCULAR_PATH
import os
import json
from tqdm import tqdm

model_path = "/mnt/beegfs/xr/models/vl/llava-v1.6-mistral-7b-hf"
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/llava/v0-20250918-180437/checkpoint-114-merged"

# mode = 'base'
# mode = 'ft'
mode = 'circular_base'
# mode = 'circular_ft'
RESULT_DIR = RESULT_DIR[mode]
os.makedirs(RESULT_DIR, exist_ok=True)
if 'circular' in mode:
    DATA_PATH = CIRCULAR_PATH


run_idx = 0
result_file = f"llava_run{run_idx}.jsonl"

new_chat_template = """{% set new_chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# MODIFICATION START: Process content sequentially #}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<image>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{# MODIFICATION END #}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '</s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}" %}"""
processor = LlavaNextProcessor.from_pretrained(model_path)
processor.tokenizer.chat_template = new_chat_template
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda")


def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_filepath).convert("RGB")

    conversation = [
        {
        "role": "user",
        "content": [
                {"type": "text", "text": ROLE_PROMPT+"Input: Image: "},
                {"type": "image"},
                {"type": "text", "text": f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"}
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    res = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

    return res

with open(DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'w+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {"image": entry['image'], "question": entry['question'], "options": entry['options'], "pred": pred, "gold": entry['answer']}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        # break