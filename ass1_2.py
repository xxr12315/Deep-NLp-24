import math
from collections import Counter
import os
def calculate_entropy(text):
    frequencies = Counter(text)
    total_length = len(text)
    entropy = 0
    for char, freq in frequencies.items():
        probability = freq / total_length
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_word_entropy(text):
    words = text.split()
    total_words = len(words)
    entropy_sum = 0
    for word in words:
        entropy_sum += calculate_entropy(word)
    return entropy_sum / total_words

def calculate_character_entropy(text):
    total_characters = len(text)
    return calculate_entropy(text) / total_characters


train_dirs = os.listdir("cibiao")
# 读取中文文本
t = ""
for name in train_dirs:
    print(name)
    with open('cibiao/'+name, 'r', encoding='gb18030') as file:
        txt = file.read()
        word_entropy = calculate_word_entropy(txt)
        print("词单位平均信息熵:", word_entropy)
        # 计算字单位的平均信息熵
        char_entropy = calculate_character_entropy(txt)
        print("字单位平均信息熵:", char_entropy)
        t = t + txt
# 示例文本

word_entropy = calculate_word_entropy(t)
print("词单位平均信息熵:", word_entropy)
# 计算字单位的平均信息熵
char_entropy = calculate_character_entropy(t)
print("字单位平均信息熵:", char_entropy)
