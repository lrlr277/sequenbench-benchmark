import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import PeftModel
import os

# --- 1. 定义路径 ---
# 原始的、未经微调的 InstructBLIP 模型路径
base_model_path = "/mnt/beegfs/xr/models/vl/instructblip-vicuna-7b"

# 你使用 Trainer 训练并保存的 LoRA 适配器权重路径
# 这应该是你的 `OUTPUT_DIR`
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--step', type=int, required=True)
args = parser.parse_args()
version = args.version
step = args.step
lora_adapter_path = f"/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/instructblip/{version}/checkpoint-{step}"

# 你想要将合并后的完整模型保存到的新路径
# merged_model_path = "../output/instructblip"
merged_model_path = f"/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/instructblip/{version}/checkpoint-{step}-merged"

# 确保输出目录存在
os.makedirs(merged_model_path, exist_ok=True)

print("开始加载基础模型和处理器...")

# --- 2. 加载基础模型和处理器 ---
# 以 bfloat16 加载以节省内存，并与你的训练设置保持一致
base_model = InstructBlipForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto", # 自动将模型分发到可用设备
)

# 处理器/分词器与基础模型一致
processor = InstructBlipProcessor.from_pretrained(base_model_path)

print(f"基础模型加载完成。模型类型: {type(base_model)}")

# --- 3. 加载 LoRA 适配器 ---
# PeftModel.from_pretrained 会将 LoRA 权重“附加”到基础模型之上
print("正在加载并附加 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
print(f"LoRA 适配器加载完成。当前模型类型: {type(model)}")


# --- 4. 合并权重并卸载适配器 ---
# 这是最关键的一步！
# .merge_and_unload() 会将 LoRA 权重合并到基础模型的权重中，
# 然后返回一个标准的、不含 peft 模块的 transformers 模型。
print("正在合并 LoRA 权重...")
model = model.merge_and_unload()
print(f"权重合并完成。合并后的模型类型: {type(model)}")


# --- 5. 保存合并后的模型 ---
# 现在你可以像保存任何标准的 Hugging Face 模型一样保存它
print(f"正在将合并后的模型保存至: {merged_model_path}")
model.save_pretrained(merged_model_path)

# 不要忘记同时保存处理器，以便后续可以方便地加载
processor.save_pretrained(merged_model_path)

print("合并后的模型和处理器已成功保存！")