from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, CIRCULAR_PATH
import os
import json
from tqdm import tqdm

model_path = "/mnt/beegfs/xr/models/vl/Phi-3.5-vision-instruct"
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/phi/v0-20250918-180609/checkpoint-114-merged"

# mode = 'base'
# mode = 'ft'
mode = 'circular_base'
# mode = 'circular_ft'
RESULT_DIR = RESULT_DIR[mode]
os.makedirs(RESULT_DIR, exist_ok=True)
if 'circular' in mode:
    DATA_PATH = CIRCULAR_PATH

run_idx = 0
result_file = f"phi_run{run_idx}.jsonl"

model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='eager'    
)

processor = AutoProcessor.from_pretrained(model_path, 
  trust_remote_code=True, 
  num_crops=4
) 

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")

    images = [image]
    placeholder = "<|image_1|>"

    prompt = ROLE_PROMPT + f"Input: Image: {placeholder}\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = processor.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 20, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, 
      eos_token_id=processor.tokenizer.eos_token_id, 
      **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=False)[0] 

    return response.strip()

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