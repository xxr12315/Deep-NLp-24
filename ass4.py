import re

import torch
import jieba
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from datasets import Dataset
# 设置torch.cuda.is_available()为False
torch.cuda.is_available = lambda: False
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

def clean_text(text):
    # 去除无法识别的字符
    cleaned_text = text.encode("ascii", "ignore").decode()
    # 去除特殊字符和重复字符
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def preprocess_chinese_corpus(file_path):
    # 加载停用词列表
    with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 分词
        words = jieba.lcut(content)
        # 去除停用词和标准化
        words = [word for word in words if word not in stop_words]
        corpus = ' '.join(words)

    return corpus


def load_dataset_from_corpus(corpus, tokenizer, model_type="t5"):
    encoded = tokenizer(corpus, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    if model_type == "t5":
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
    elif model_type == "gpt2":
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        labels = encoded["input_ids"].clone()
    else:
        raise ValueError("Unsupported model type")

    dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "labels": labels.tolist()
    })
    return dataset


# 通用的微调函数
def finetune_model(model, train_dataset, output_dir, tokenizer, model_type="t5"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    if model_type == "t5":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    elif model_type == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        raise ValueError("Unsupported model type")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()


# 加载预训练模型和分词器
t5_model_path = "./models/t5-base-chinese-cluecorpussmall"
gpt2_model_path = "./models/gpt2-chinese-cluecorpussmall"

# 加载模型和分词器
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(torch.device('cpu'))
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path, use_fast=False)

gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(torch.device('cpu'))
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path, vocab_file='./models/gpt2-chinese-cluecorpussmall/vocab.json')


gpt2_embedding_weights = gpt2_model.transformer.wte.weight
vocab = gpt2_tokenizer.get_vocab()
# 预处理中文语料库
corpus_file = 'out.txt'
chinese_corpus = preprocess_chinese_corpus(corpus_file)

# 加载数据集
train_dataset_t5 = load_dataset_from_corpus(chinese_corpus, t5_tokenizer, model_type="t5")
train_dataset_gpt2 = load_dataset_from_corpus(chinese_corpus, gpt2_tokenizer, model_type="gpt2")

# 模型微调
finetune_model(t5_model, train_dataset_t5, "./t5-finetuned", t5_tokenizer)
finetune_model(gpt2_model, train_dataset_gpt2, "./gpt2-finetuned", gpt2_tokenizer, model_type="gpt2")


def generate_text(model, tokenizer, prompt, model_type, max_length=50, num_beams=20):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    if model_type == "t5":
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=num_beams,
            attention_mask=inputs.attention_mask,
            early_stopping=True
        )
    elif model_type == "gpt2":
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    else:
        raise ValueError("Unsupported model type")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 后处理步骤
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for p in punctuation:
        generated_text = generated_text.replace(p * 3, p)

    generated_text = ' '.join(generated_text.split())
    return generated_text


prompt = "武林至尊"
generated_text_t5 = generate_text(t5_model, t5_tokenizer, prompt, model_type="t5")
print(f"Generated text by Seq2Seq (T5): {generated_text_t5}")

generated_text_gpt2 = generate_text(gpt2_model, gpt2_tokenizer, prompt, model_type="gpt2")
print(f"Generated text by Transformer (GPT-2): {generated_text_gpt2}")






