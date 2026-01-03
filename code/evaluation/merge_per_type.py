import json
import pandas as pd
import numpy as np

# 读取两个结果
with open("output/base.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)
with open("output/ft.json", "r", encoding="utf-8") as f:
    ft_data = json.load(f)
with open("output/close.json", "r", encoding="utf-8") as f:
    close_data = json.load(f)
with open("output/other.json", "r", encoding="utf-8") as f:
    other_data = json.load(f)
csv_path = f"output/acc_per_type.csv"
rows = []

# 提取四个指标：weighted P/R/F1 + acc
def extract(metrics):
    return (m['acc'] for m in metrics["acc_per_type"].values())

set_rows = []
for model in ft_data.keys():
    accs = extract(ft_data[model])
    rows.append([model, "Lora", *accs])
    set_rows.append(accs)

# if set_rows:  # 防止为空
#     mean_accs = np.mean(set_rows, axis=0)
#     mean_accs = [round(x, 2) for x in mean_accs]
#     rows.append(["Average", "-", *mean_accs])

rows.append([])
set_rows = []
close_models = ['gpt5', 'gemini']
for model in close_models:
    close_0shot = extract(close_data[f"{model}_0shot"])
    close_3shot = extract(close_data[f"{model}_3shot"])

    rows.append([model, "0-shot", *close_0shot])
    rows.append(['', "few-shot", *close_3shot])

rows.append([])
other_models = ['random', 'human', 'human_text_only']
for model in other_models:
    accs = extract(other_data[model])
    if model.endswith("text_only"):
        rows.append([model, "text-only", *accs])
    else:
        rows.append([model, "-", *accs])


# 转换为 DataFrame
df = pd.DataFrame(rows, columns=["Model", "Set."] + [str(i+1) for i in range(9)])

df.to_csv(csv_path, index=False, encoding="utf-8")
