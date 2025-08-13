import os
import json
def load_training_history(path):
    """加载历史训练数据用于断点续训"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)
        return history
    return None

history = load_training_history('./tmp/train/training_history.json')
# print(history['test_loss'])
# print(len(history['test_loss']))
for i in history:
    print(i, len(history[i]))