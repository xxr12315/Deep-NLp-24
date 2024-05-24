import sys

import jieba
import random
from gensim.models import Word2Vec
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 准备数据和中文分词
def preprocess_chinese_corpus(folder_path):
    corpus = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='gb18030') as f:
                content = f.read()
                tokenized_sentences = [jieba.lcut(sentence) for sentence in content.split('\n') if sentence.strip()]
                corpus.extend(tokenized_sentences)
    return corpus

chinese_tokenized_sentences = preprocess_chinese_corpus('cibiao')


# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=chinese_tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec.model")



# 计算语义距离
def compute_similarity(model, word1, word2):
    if isinstance(model, Word2Vec):
        vector1 = model.wv[word1]
        vector2 = model.wv[word2]
    else:
        vector1 = model.word_vectors[model.dictionary[word1]]
        vector2 = model.word_vectors[model.dictionary[word2]]
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

# 从词汇表中随机选择两个词语
word1 = "刀光"
word2 = "剑影"

word2vec_similarity = compute_similarity(word2vec_model, word1, word2)


print(f"Word1: {word1}, Word2: {word2}")
print(f"Word2Vec similarity between '{word1}' and '{word2}': {word2vec_similarity}")

word1 = "猫"
word2 = "狗"

word2vec_similarity = compute_similarity(word2vec_model, word1, word2)


print(f"Word1: {word1}, Word2: {word2}")
print(f"Word2Vec similarity between '{word1}' and '{word2}': {word2vec_similarity}")

word1 = "牛肉"
word2 = "剑影"

word2vec_similarity = compute_similarity(word2vec_model, word1, word2)


print(f"Word1: {word1}, Word2: {word2}")
print(f"Word2Vec similarity between '{word1}' and '{word2}': {word2vec_similarity}")

# 词语聚类
def plot_clusters(model, num_clusters=5, filename='clusters.png'):
    if isinstance(model, Word2Vec):
        words = list(model.wv.index_to_key)
        word_vectors = model.wv[words]
    else:
        words = list(model.dictionary.keys())
        word_vectors = model.word_vectors

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
    labels = kmeans.labels_

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = word_vectors[i][0], word_vectors[i][1]
        plt.scatter(x, y, c=f'C{label}')
        plt.text(x+0.03, y+0.03, words[i], fontsize=9)
    plt.savefig(filename)
    plt.show()

# 可视化Word2Vec聚类结果
plot_clusters(word2vec_model)

# 计算段落语义关联
def get_paragraph_vector(paragraph, model):
    tokens = [word for word in jieba.lcut(paragraph) if word in model.wv] if isinstance(model, Word2Vec) else [word for word in jieba.lcut(paragraph) if word in model.dictionary]
    vectors = [model.wv[token] for token in tokens] if isinstance(model, Word2Vec) else [model.word_vectors[model.dictionary[token]] for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 从语料库中随机选择两个句子作为示例段落
sample_paragraph1 = random.choice(chinese_tokenized_sentences)
sample_paragraph2 = random.choice(chinese_tokenized_sentences)

# 计算示例段落的语义关联
sample_paragraph1_str = ' '.join(sample_paragraph1)
sample_paragraph2_str = ' '.join(sample_paragraph2)
paragraph_vector1 = get_paragraph_vector(sample_paragraph1_str, word2vec_model)
paragraph_vector2 = get_paragraph_vector(sample_paragraph2_str, word2vec_model)

paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
print(f"Semantic similarity between paragraphs: {paragraph_similarity}")
print(sample_paragraph1_str)

print(sample_paragraph2_str)




# 计算示例段落的语义关联
sample_paragraph1_str = '文本标注子系统的功能目标为提供一个能够完成文本标注与管理的图形化界面平台，以直接提供可供科研使用的有效标注数据或为智能挖掘模型训练提供训练数据'
sample_paragraph2_str = '标注管理员创建标注项目，填写项目相关信息，加入标注文本，指定标注人员，添加标签后为标注人员分配任务'
paragraph_vector1 = get_paragraph_vector(sample_paragraph1_str, word2vec_model)
paragraph_vector2 = get_paragraph_vector(sample_paragraph2_str, word2vec_model)

paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
print(f"Semantic similarity between paragraphs: {paragraph_similarity}")
print(sample_paragraph1_str)

print(sample_paragraph2_str)