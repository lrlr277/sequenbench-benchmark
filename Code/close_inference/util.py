import os
import random
import base64
import jsonlines
import json
random.seed(42)

TEST_DATA_PATH = r"D:\归档\科研\2508-多模态实验\seq\data\split_712\test_200.jsonl"  
# TEST_DATA_PATH = r"D:\归档\科研\2508-多模态实验\seq\data\split_73\test_200_circular.jsonl"
RERUN_MODES = ["gemini_0shot", "gemini_1shot", "gpt5_0shot", "gpt5_2shot"]
RERUN_DATA_DICT = {mode: rf"D:\归档\科研\2508-多模态实验\seq\data\split_712\test_rerun_100_circular_{mode}.jsonl" for mode in RERUN_MODES}
TRAIN_DATA_PATH = r"D:\归档\科研\2508-多模态实验\seq\data\split_712\train.jsonl"  
IMAGE_DIR = r"D:\归档\科研\2508-多模态实验\seq\data\image"  
RESULT_DIR = r"D:\归档\科研\2508-多模态实验\seq\model_output\close"
# RESULT_DIR = r"D:\归档\科研\2508-多模态实验\seq\model_output\circular_close"
# RESULT_DIR = r"D:\归档\科研\2508-多模态实验\seq\model_output\close_rerun"
# RESULT_DIR = r"D:\归档\科研\2508-多模态实验\seq\model_output\circular_close_rerun"
ICL_PATH = r"D:\归档\科研\2508-多模态实验\seq\inference\icl"
ROLE_PROMPT = """
You are currently a senior expert in sequence problems, focusing on specific research topics such as time, space, length, quantity, emotion, symmetry, logic, etc. You can sort their attributes, such as length, order, size, quantity, and strength. Given one or more Images and a sorting problem, some questions can be answered directly, some questions require inference, and others can only choose relatively reasonable answers. Your task is to answer these questions from a human perspective and output the correct answers without any explanation. Please note that you only need to select one option from all options, and your response should only include the option letter (A, B, C, or D) and not any other text.
""".strip()+'\n'

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data
train_data = read_jsonl(TRAIN_DATA_PATH)

def encode_image_to_base64(path):
  with open(path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def init_json(file_path):
    with open(file_path, 'w') as f:
        json.dump({}, f, indent=2)

def construct_few_shot(entry, num_shot, model, mode="random"):
    global train_data

    # 对图片相同的问题取相同的ICL例子 用于circular模式
    icl_file = os.path.join(ICL_PATH, f'{model}_{num_shot}shot.json')
    if not os.path.exists(icl_file):
        init_json(icl_file)
    with open(icl_file, 'r') as f:
        icl_dict = json.load(f)

    img_path = os.path.join(IMAGE_DIR, entry['image'])
    base64_image = encode_image_to_base64(img_path)
    test_content = [
        {"type": "text", "text": "Input: Image: "},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        {"type": "text", "text": f"\nQuestion: {entry['question']}, Options: {'; '.join(entry['options'])}.\nOutput:"}
    ]
    
    fewshot_prompt = f"Given the following {num_shot} examples to learn the spatial relation reasoning task:\n" if num_shot else ""
    content = [
        {"type": "text", "text": ROLE_PROMPT + fewshot_prompt},
    ]

    if mode == "random":
        if entry['image'] in icl_dict:
            sampled_data = icl_dict[entry['image']]
        else:
            sampled_data = random.sample(train_data, num_shot)
            icl_dict[entry['image']] = sampled_data
    else:
        sampled_data = []
    
    with open(icl_file, 'w') as f:
        json.dump(icl_dict, f, indent=2)

    if len(sampled_data) != num_shot:
        raise ValueError("len(sampled_data) != num_shot")
    else:
        fewshot_content = []
        for idx, example in enumerate(sampled_data):
            ex_img_path = os.path.join(IMAGE_DIR, example['image'])
            ex_base64_image = encode_image_to_base64(ex_img_path)
            ex_answer_str = example['options'][ord(example['answer']) - ord('A')]
            fewshot_content.extend([
                {"type": "text", "text": f"Example{idx+1}:\nInput: Image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_base64_image}"}},
                {"type": "text", "text": f"\nQuestion: {example['question']}, Options: {'; '.join(example['options'])}.\nOutput: {ex_answer_str}\n"}
            ])
    content.extend(fewshot_content)
    content.extend(test_content)
    return content