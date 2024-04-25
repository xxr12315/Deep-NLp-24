import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import os


# 函数：从语料库中抽取数据集
def extract_dataset(corpus, labels, num_paragraphs):
    # 计算每个标签需要抽取的段落数量
    num_paragraphs_per_label = num_paragraphs // len(set(labels))
    dataset = []
    dataset_labels = []
    for label in set(labels):
        # 从具有特定标签的段落中均匀抽取指定数量的段落
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, labels) if paragraph_label == label]
        sampled_paragraphs = np.random.choice(label_paragraphs, num_paragraphs_per_label, replace=False)
        dataset.extend(sampled_paragraphs)
        dataset_labels.extend([label] * num_paragraphs_per_label)
    return dataset, dataset_labels


# 主函数
def main():
    # 读取语料库，假设语料库是一个包含小说段落的文件，每行为一个段落
    folder_path = "cibiao"

    # 读取文件夹下所有txt文件的内容并合并成一个语料库
    corpus = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r",encoding='gb18030') as file:
                text = file.read()
                paragraphs = text.split("\n")
                corpus.extend(paragraphs)
                labels.extend(["novel_" + file_name.split(".")[0]] * len(paragraphs))

    # 定义不同的 K
    K_values = [20, 100, 500, 1000, 3000]

    # 定义不同的主题数量 T
    T_values = [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 500, 1000, 3000]

    # 定义交叉验证的次数
    num_cross_val = 10

    # 定义分类器
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Multinomial Naive Bayes": MultinomialNB()
    }

    # 定义结果存储列表
    results = []

    # 遍历不同的 K 和 T
    for K in K_values:
        for T in T_values:
            # 抽取数据集
            dataset, dataset_labels = extract_dataset(corpus, labels, num_paragraphs=1000)

            # 将数据集划分为训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels, test_size=0.1, random_state=42)

            # 将文本转换为主题分布的流水线
            lda_pipeline = Pipeline([
                ('vectorizer', CountVectorizer(max_features=K, analyzer='word')),
                ('lda', LatentDirichletAllocation(n_components=T, random_state=42))
            ])

            # 将文本转换为主题分布
            X_train_lda = lda_pipeline.fit_transform(X_train)
            X_test_lda = lda_pipeline.transform(X_test)

            # 使用不同的分类器进行训练和评估
            for classifier_name, classifier in classifiers.items():
                # 保存结果

                classifier.fit(X_train_lda, y_train)
                accuracy = np.mean(cross_val_score(classifier, X_train_lda, y_train, cv=num_cross_val))
                test_accuracy = accuracy_score(y_test, classifier.predict(X_test_lda))

                # 保存结果

                results.append({
                    'K': K,
                    'T': T,
                    'Classifier': classifier_name,
                    'Analyzer': 'Word',
                    'Training Accuracy': accuracy,
                    'Test Accuracy': test_accuracy
                })

            # 将文本转换为主题分布的流水线（以字为基本单元）
            lda_pipeline_char = Pipeline([
                ('vectorizer', CountVectorizer(max_features=K, analyzer='char')),
                ('lda', LatentDirichletAllocation(n_components=T, random_state=42))
            ])

            # 将文本转换为主题分布
            X_train_lda_char = lda_pipeline_char.fit_transform(X_train)
            X_test_lda_char = lda_pipeline_char.transform(X_test)

            # 使用不同的分类器进行训练和评估
            for classifier_name, classifier in classifiers.items():
                classifier.fit(X_train_lda_char, y_train)
                accuracy = np.mean(cross_val_score(classifier, X_train_lda_char, y_train, cv=num_cross_val))
                test_accuracy = accuracy_score(y_test, classifier.predict(X_test_lda_char))

                # 保存结果
                results.append({
                    'K': K,
                    'T': T,
                    'Classifier': classifier_name,
                    'Analyzer': 'Char',
                    'Training Accuracy': accuracy,
                    'Test Accuracy': test_accuracy
                })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果到xlsx文件
    results_df.to_excel("result3s.xlsx", index=False)
if __name__ == "__main__":
    main()