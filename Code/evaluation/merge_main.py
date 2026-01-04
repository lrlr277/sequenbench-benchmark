import json
import pandas as pd

# 读取两个结果
with open("output/base.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)
with open("output/ft.json", "r", encoding="utf-8") as f:
    ft_data = json.load(f)
with open("output/close.json", "r", encoding="utf-8") as f:
    close_data = json.load(f)
with open("output/other.json", "r", encoding="utf-8") as f:
    other_data = json.load(f)
csv_path = f"output/main.csv"
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

# 按照 base.json 的模型顺序遍历
for model in base_data.keys():
    base_metrics = base_data[model]
    ft_metrics = ft_data.get(model, {})  # 如果ft里没有就留空
    
    base_p, base_r, base_f1, base_acc = extract(base_metrics)
    ft_p, ft_r, ft_f1, ft_acc = extract(ft_metrics)
    
    rows.append([
        model,
        "Base", base_p, base_r, base_f1, base_acc,
        "Lora", ft_p, ft_r, ft_f1, ft_acc,
    ])

rows.append([])
close_models = ['gpt5', 'gemini']
for model in close_models:
    close_metrics = []
    close_metrics.append(close_data[f"{model}_0shot"])
    close_metrics.append(close_data[f"{model}_1shot"])
    close_metrics.append(close_data[f"{model}_2shot"])
    close_metrics.append(close_data[f"{model}_3shot"])

    close_results = []
    for metric in close_metrics:
        close_results.append(extract(metric))
    
    rows.append([
        model,
        "0shot", close_results[0][0], close_results[0][1], close_results[0][2], close_results[0][3],
        "2shot", close_results[2][0], close_results[2][1], close_results[2][2], close_results[2][3],
    ])

    rows.append([
        model,
        "1shot", close_results[1][0], close_results[1][1], close_results[1][2], close_results[1][3],
        "3shot", close_results[3][0], close_results[3][1], close_results[3][2], close_results[3][3],
    ])

rows.append([])
other_models = ['random', 'human']
for model in other_models:
    base_metrics = other_data[model]
    if model == 'human':
        ft_metrics = other_data.get(f"{model}_text_only", {})  # 如果ft里没有就留空
        
        base_p, base_r, base_f1, base_acc = extract(base_metrics)
        ft_p, ft_r, ft_f1, ft_acc = extract(ft_metrics)

        rows.append([
            model,
            "-", base_p, base_r, base_f1, base_acc,
            "Text-only", ft_p, ft_r, ft_f1, ft_acc,
        ])
    else:
        base_p, base_r, base_f1, base_acc = extract(base_metrics)

        rows.append([
            model,
            "-", base_p, base_r, base_f1, base_acc,
        ])

# 转换为 DataFrame
df = pd.DataFrame(rows, columns=[
    "Model", "Set.", "P", "R", "F1", "Acc",
    "Set.", "P", "R", "F1", "Acc"
])

df.to_csv(csv_path, index=False, encoding="utf-8")
