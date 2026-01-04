from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, CIRCULAR_PATH
import os
import json
from tqdm import tqdm

# base
# model_path = "/mnt/beegfs/xr/models/vl/Qwen2.5-VL-7B-Instruct"
# model_path = "/mnt/beegfs/xr/models/vl/Qwen2.5-VL-32B-Instruct"
model_path = "/mnt/beegfs/xr/models/vl/Qwen2.5-VL-72B-Instruct"

# ft
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/qwen_7b/checkpoint-114-merged"
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/qwen_32b/checkpoint-114-merged"
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/qwen_72b/v1-20250921-125716/checkpoint-120-merged"

# mode = 'base'
mode = 'ft'
# mode = 'circular_base'
# mode = 'circular_ft'
RESULT_DIR = RESULT_DIR[mode]
os.makedirs(RESULT_DIR, exist_ok=True)
if 'circular' in mode:
    DATA_PATH = CIRCULAR_PATH

run_idx = 0
# result_file = f"qwen_7b_run{run_idx}.jsonl"
# result_file = f"qwen_32b_run{run_idx}.jsonl"
result_file = f"qwen_72b_run{run_idx}.jsonl"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_path)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ROLE_PROMPT+"Input: Image: "},
                {
                    "type": "image",
                    "image": image_filepath,
                },
                {"type": "text", "text": f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

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