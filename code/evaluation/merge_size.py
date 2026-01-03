import json
import pandas as pd

# 读取两个结果
with open("output/base.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)
with open("output/ft.json", "r", encoding="utf-8") as f:
    ft_data = json.load(f)
csv_path = f"output/size.csv"
rows = []

# 提取四个指标：weighted P/R/F1 + acc
def extract(metrics):
    if not metrics:
        return (None, None, None, None)
    return (
        metrics["p"],
        metrics["r"],
        metrics["f1"],
        metrics["acc"],
    )

models = ['qwen_7b', 'qwen_32b', 'qwen_72b', 'intern_8b', 'intern_14b', 'intern_38b', 'intern_78b', 'qwen3_8b', 'qwen3_32b', 'intern3_5_8b', 'intern3_5_14b', 'intern3_5_38b']
for model in models:
    base_metrics = base_data[model]
    ft_metrics = ft_data.get(model, {})  # 如果ft里没有就留空
    
    base_p, base_r, base_f1, base_acc = extract(base_metrics)
    ft_p, ft_r, ft_f1, ft_acc = extract(ft_metrics)
    
    rows.append([
        model,
        "Base", base_p, base_r, base_f1, base_acc,
        "Lora", ft_p, ft_r, ft_f1, ft_acc,
    ])

# 转换为 DataFrame
df = pd.DataFrame(rows, columns=[
    "Model", "Set.", "P", "R", "F1", "Acc",
    "Set.", "P", "R", "F1", "Acc"
])

df.to_csv(csv_path, index=False, encoding="utf-8")
