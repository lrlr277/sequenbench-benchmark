import os
import json

input_file = "test.jsonl"     # 原始 jsonl 文件
output_dir = "parts"           # 输出目录
num_parts = 14                  # 分成 14 份

os.makedirs(output_dir, exist_ok=True)

# 读取全部行
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
assert total == 700, f"文件不是700行，而是 {total} 行"

part_size = total // num_parts  # 50

print(f"总行数: {total}, 每份: {part_size} 行")

# 按顺序切分并保存（从0开始命名）
for i in range(num_parts):
    start = i * part_size
    end = (i + 1) * part_size

    part_lines = lines[start:end]

    part_file = os.path.join(output_dir, f"part_{i}.jsonl")
    with open(part_file, "w", encoding="utf-8") as f:
        f.writelines(part_lines)

    print(f"Saved: {part_file}  ({start} - {end - 1})")
