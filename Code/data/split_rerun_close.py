import jsonlines
import random

n = 100
test_file = "split_73/test.jsonl"
close_file = f"split_73/test_rerun_{n}.jsonl"
ref_file = f"split_73/test_200.jsonl"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    print(len(data), data[0].keys())
    return data

def to_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)

def stat_num(data):
    """统计总体、选项分布、大类分布"""
    stat = {}
    stat['all'] = len(data)

    # 选项统计
    for opt in ['a','b','c','d']:
        stat[opt] = len([d for d in data if d['answer'].lower()==opt])

    # 大类统计（image 第一位字符）
    for cat in [str(i) for i in range(10)]:
        stat[cat] = len([d for d in data if str(d['image'])[0] == cat])

    return stat

def get_ratio(stat):
    """转为比例"""
    total = stat['all']
    ratio = {}
    for k, v in stat.items():
        if k != 'all':
            ratio[k] = v / total if total > 0 else 0
    return ratio

def sample_with_tolerance(data, n, tolerance=0.01, max_retry=10000):
    """抽样，保证选项和大类比例误差在 tolerance 内"""
    full_stat = stat_num(data)
    full_ratio = get_ratio(full_stat)

    for attempt in range(max_retry):
        sample = random.sample(data, n)
        sample_stat = stat_num(sample)
        sample_ratio = get_ratio(sample_stat)

        diffs = [abs(sample_ratio[k] - full_ratio[k]) for k in full_ratio.keys()]
        if all(diff <= tolerance for diff in diffs):
            print(f"成功抽样: 尝试 {attempt+1} 次")
            print("总体分布:", full_ratio)
            print("抽样分布:", sample_ratio)
            return sample

    raise RuntimeError("超过最大尝试次数，未能找到满足条件的抽样")

all_data = read_jsonl(test_file)
ref_data = read_jsonl(ref_file)
print(len(all_data), len(ref_data))
all_data = [d for d in all_data if d['id'] not in [r['id'] for r in ref_data]]
print(len(all_data))

close_data = sample_with_tolerance(all_data, n, tolerance=0.05)

stat = stat_num(close_data)
print(stat)

to_jsonl(close_data, close_file)

# 总体分布: {'a': 0.2357142857142857, 'b': 0.25285714285714284, 'c': 0.24, 'd': 0.2714285714285714, '0': 0.0, '1': 0.20714285714285716, '2': 0.08571428571428572, '3': 0.19714285714285715, '4': 0.2714285714285714, '5': 0.10142857142857142, '6': 0.03571428571428571, '7': 0.041428571428571426, '8': 0.04285714285714286, '9': 0.017142857142857144}
# 抽样分布: {'a': 0.235, 'b': 0.255, 'c': 0.245, 'd': 0.265, '0': 0.0, '1': 0.205, '2': 0.08, '3': 0.2, '4': 0.28, '5': 0.105, '6': 0.04, '7': 0.035, '8': 0.04, '9': 0.015}
# {'all': 200, 'a': 47, 'b': 51, 'c': 49, 'd': 53, '0': 0, '1': 41, '2': 16, '3': 40, '4': 56, '5': 21, '6': 8, '7': 7, '8': 8, '9': 3}