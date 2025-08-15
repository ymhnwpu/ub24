import json

# 读取training_history.json文件
with open('./tmp/train/training_history.json', 'r') as f:
    history = json.load(f)

# 打印各个指标的原始长度
print("Original length of each metric in training_history.json:")
for key, value in history.items():
    print(f"{key}: {len(value)}")

# 仅保留前55个元素（下标0到54），并写会文件
# print("\nAfter keeping only the first 55 elements (index 0-54):")
# for key, value in history.items():
#     history[key] = value[:55]
#     print(f"{key}: {len(history[key])}")
# with open('./tmp/train/training_history.json', 'w') as f:
#     json.dump(history, f)
# print("\ntraining_history.json has been updated with only the first 55 elements for each metric.")