import sys

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import jieba
import os

def zipf_law(text):
    # 分词
    words = list(jieba.cut(text))

    # 计算词频
    word_freq = Counter(words)

    # 根据词频排序
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # 提取词频和排名
    freq = [item[1] for item in sorted_word_freq]
    rank = np.arange(1, len(freq) + 1)

    # 绘制频率与排名的对数图
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(rank), np.log(freq), marker='o', linestyle='')
    plt.title("Zipf's Law Verification")
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.grid(True)
    plt.show()


train_dirs = os.listdir("cibiao")
txt = ""
# 读取中文文本
for name in train_dirs:
    print(name)
    with open('cibiao/'+name, 'r', encoding='gb18030') as file:
        txt = txt + file.read()


zipf_law(txt)