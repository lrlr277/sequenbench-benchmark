DATA_PATH = "/mnt/beegfs/xr/lm_multimodal/seq/data/split_712/test.jsonl"  
DATA_PART_PATH = [f"/mnt/beegfs/xr/lm_multimodal/seq/data/split_712/parts/part_{str(i)}.jsonl" for i in range(14)]
CIRCULAR_PATH = "/mnt/beegfs/xr/lm_multimodal/seq/data/split_712/test_circular.jsonl"  
IMAGE_DIR = "/mnt/beegfs/xr/lm_multimodal/seq/data/images"  
RESULT_DIR = {
    "base": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/base",
    "ft": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/ft",
    "test": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/test",
    "circular_base": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/circular_base",
    "circular_ft": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/circular_ft",
    "cot_base": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/cot_base",
    "cot_ft": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/cot_ft",
    "other": "/mnt/beegfs/xr/lm_multimodal/seq/model_output/other"
}
ROLE_PROMPT = """
You are currently a senior expert in sequence problems, focusing on specific research topics such as time, space, length, quantity, emotion, symmetry, logic, etc. You can sort their attributes, such as length, order, size, quantity, and strength. Given one or more images and a sorting problem, some questions can be answered directly, some questions require inference, and others can only choose relatively reasonable answers. Your task is to answer these questions from a human perspective and output the correct answers without any explanation. Please note that you only need to select one option from all options, and your response should only include the option letter (A, B, C, or D) and not any other text.
""".strip()+'\n'

COT_ROLE_PROMPT = """
You are currently a senior expert in sequence problems, focusing on specific research topics such as time, space, length, quantity, emotion, symmetry, logic, etc. You can sort their attributes, such as length, order, size, quantity, and strength. Given one or more images and a sorting problem, some questions can be answered directly, some questions require inference, and others can only choose relatively reasonable answers. Your task is to answer these questions from a human perspective and output the correct answers. Please note that you only need to select one option from all options.
""".strip()+'\n'