import torch
from transformers import AutoModelForCausalLM

from janus.models import VLChatProcessor, MultiModalityCausalLM
from janus.utils.io import load_pil_images

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, CIRCULAR_PATH
import os
import json
from tqdm import tqdm

model_path = "/mnt/beegfs/xr/models/vl/Janus-Pro-7B"
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--version', type=str, required=True)
# parser.add_argument('--step', type=int, required=True)
# args = parser.parse_args()
# version = args.version
# step = args.step
# model_path = f"/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/janus/{version}/checkpoint-{step}-merged"
# model_path = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/janus/v0-20250918-105049/checkpoint-40-merged"

# mode = 'base'
# mode = 'ft'
mode = 'circular_base'
# mode = 'circular_ft'
RESULT_DIR = RESULT_DIR[mode]
os.makedirs(RESULT_DIR, exist_ok=True)
if 'circular' in mode:
    DATA_PATH = CIRCULAR_PATH

run_idx = 0
result_file = f"janus_run{run_idx}.jsonl"
# result_file = f"janus_ckpt{step}.jsonl"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    prompt = ROLE_PROMPT + f"Input: Image: <image_placeholder>\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
    conversation = [
        {
            "role": "User",
            "content": prompt,
            "Images": [image_filepath]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # load Images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=500,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
    return answer

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