import json
import re
from pathlib import Path

# 输入文件路径（你要替换成你的json文件路径）
json_file = "/mnt/beegfs/xr/lm_multimodal/seq/evaluation/output/test.json"

# 输出日志文件
log_file = Path("/mnt/beegfs/xr/lm_multimodal/seq/evaluation/output/test.log")
order_file = Path("/mnt/beegfs/xr/lm_multimodal/seq/evaluation/model_order_test.txt")

# 读取 JSON
with open(json_file, "r") as f:
    data = json.load(f)

# 按 model 分类
models = {}
for key, val in data.items():
    m = re.match(r"(.+)_ckpt(\d+)", key)
    if not m:
        continue
    model, ckpt = m.group(1), int(m.group(2))
    final_f1 = val.get("f1", 0)
    acc = val.get("acc", 0)

    if model not in models:
        models[model] = []
    models[model].append((ckpt, final_f1, acc))

# 逐个 model 输出
with open(log_file, "a") as f:
    for model, entries in models.items():
        f.write(f"\n{model}\n")
        # ckpt 升序
        entries.sort(key=lambda x: x[0])
        # 找最高 acc
        max_acc = max(entries, key=lambda x: x[2])[2]
        for ckpt, f1, acc in entries:
            f1_str = f"{int(f1*10000):04d}"[:4]
            acc_str = f"{int(acc*10000):04d}"[:4]
            line = f"{ckpt} {f1_str},{acc_str}"
            if abs(acc - max_acc) < 1e-12:  # 避免浮点误差
                line += "**"
            f.write(line + "\n")

with open(order_file, "w") as f:
    f.write("")