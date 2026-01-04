import json
import pandas as pd

# 读取两个结果
with open("output/base.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)
with open("output/ft.json", "r", encoding="utf-8") as f:
    ft_data = json.load(f)
with open("output/close.json", "r", encoding="utf-8") as f:
    close_data = json.load(f)
with open("output/circular_base.json", "r", encoding="utf-8") as f:
    circular_base_data = json.load(f)
with open("output/circular_ft.json", "r", encoding="utf-8") as f:
    circular_ft_data = json.load(f)
with open("output/circular_close.json", "r", encoding="utf-8") as f:
    circular_close_data = json.load(f)
csv_path = f"output/circular.csv"
rows = []

def extract(metrics):
    if not metrics:
        return None
    return metrics["acc"]

for model in circular_base_data.keys():
    base_metrics = base_data[model]
    ft_metrics = ft_data[model]
    circular_base_metrics = circular_base_data[model]
    circular_ft_metrics = circular_ft_data[model]
    
    base_acc = extract(base_metrics)
    ft_acc = extract(ft_metrics)
    circular_base_acc = extract(circular_base_metrics)
    circular_ft_acc = extract(circular_ft_metrics)
    base_change = circular_base_acc - base_acc
    ft_change = circular_ft_acc - ft_acc
    
    rows.append([
        model,
        "Base", base_acc, circular_base_acc, base_change,
        "Lora", ft_acc, circular_ft_acc, ft_change,
    ])

rows.append([])
models = ['gpt5', 'gemini']
for model in models:
    zero = extract(close_data[f"{model}_0shot"])
    three = extract(close_data[f"{model}_3shot"])
    circular_zero = extract(circular_close_data[f"{model}_0shot"])
    circular_three = extract(circular_close_data[f"{model}_3shot"])
    
    zero_change = circular_zero - zero
    three_change = circular_three - three
    
    rows.append([
        model,
        "0-shot", zero, circular_zero, zero_change,
        "3-shot", three, circular_three, three_change,
    ])

# 转换为 DataFrame
df = pd.DataFrame(rows, columns=[
    "Model", "Set.", "Vanilla", "Circular", "Acc Change",
    "Set.", "Vanilla", "Circular", "Acc Change"
])

df.to_csv(csv_path, index=False, encoding="utf-8")
