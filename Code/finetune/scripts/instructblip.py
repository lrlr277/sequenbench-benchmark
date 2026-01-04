import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# --- 1. 配置模型和数据路径 (请根据你的实际情况修改) ---
# 基础模型路径
MODEL_PATH = "/mnt/beegfs/xr/models/vl/instructblip-vicuna-7b"
TRAIN_DATA_PATH = "/mnt/beegfs/xr/lm_multimodal/seq/data/split_712/train_sharegpt_instructblip.jsonl"
# 微调后模型的输出路径
OUTPUT_DIR = "/mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/instructblip/v0"

# --- 2. 自定义数据集类 (与之前相同) ---
class InstructBlipDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_length=512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = entry['Images'][0]
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return None # 或者返回一个占位符

        prompt = entry['messages'][0]['content']
        answer = entry['messages'][1]['content']
        
        # 预处理文本和图像
        encoding = self.processor(
            images=image, 
            text=prompt, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 预处理标签
        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=20 # 答案通常很短
        ).input_ids
        
        # 将 padding token 设置为 -100 以便在计算损失时忽略
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        encoding["labels"] = labels.squeeze()
        # 将所有张量从批次维度中挤压出来，因为 collator 会重新添加它
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# --- 3. 加载模型和处理器 ---
processor = InstructBlipProcessor.from_pretrained(MODEL_PATH)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, # 使用 bfloat16 以获得更好的性能和稳定性
    trust_remote_code=True # 对齐 --trust_remote_code True
)

# --- 4. LoRA 配置 (与 llamafactory-cli 对齐) ---
config = LoraConfig(
    r=32,                   # --lora_rank 32
    lora_alpha=16,          # --lora_alpha 16
    lora_dropout=0.0,       # --lora_dropout 0
    bias="none",
    # --lora_target all 的一个合理解释是针对所有注意力模块的线性层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# --- 5. 准备数据集 ---
train_dataset = InstructBlipDataset(
    jsonl_path=TRAIN_DATA_PATH,
    processor=processor,
)

# --- 6. 定义训练参数 (与 llamafactory-cli 对齐) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    num_train_epochs=6,                     # --num_train_epochs 6
    per_device_train_batch_size=8,          # --per_device_train_batch_size 8
    gradient_accumulation_steps=16,         # --gradient_accumulation_steps 16
    learning_rate=2e-5,                     # --learning_rate 2e-5
    lr_scheduler_type="cosine",             # --lr_scheduler_type cosine
    optim="adamw_torch",                    # --optim adamw_torch
    
    logging_steps=20,                       # --logging_steps 10
    save_strategy="steps",                  # 设置保存策略为 steps
    save_steps=20,                          # --save_steps 20
    
    bf16=True,                              # 使用 bfloat16, 现代 GPU 上比 fp16 更稳定
    remove_unused_columns=False,
    dataloader_num_workers=16,              # --preprocessing_num_workers 16
    
    load_best_model_at_end=False,            # 训练结束时加载最佳模型
)

# --- 7. 初始化 Trainer ---
# 使用 DataCollatorForSeq2Seq 来正确处理标签的填充
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    label_pad_token_id=-100, # 确保标签填充符被正确忽略
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# --- 8. 开始训练 ---
print("开始 LoRA 微调 (参数已与 llamafactory-cli 示例对齐)...")
trainer.train()

# --- 9. 保存最终模型 ---
print("训练完成，保存最终的 LoRA 适配器...")
trainer.save_model(OUTPUT_DIR) # trainer.save_model 会保存适配器权重
processor.save_pretrained(OUTPUT_DIR)
print(f"模型已保存至: {OUTPUT_DIR}")