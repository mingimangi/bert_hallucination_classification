# Detecting Hallucinations in Question-Answering System

This repository contains a simple but effective solution for detecting hallucinations in generated text, leveraging the power of transformer-based language models. Hallucinations, in this context, refer to incorrect or misleading information produced by models when generating responses to given queries. I frame the task of hallucination detection as a classification problem using a pre-trained BERT model. Fine-tuning a pre-trained BERT model on my specific dataset benefits from leveraging the extensive language understanding BERT gained from being pre-trained on large amounts of diverse text data, enhancing its ability to accurately detect hallucinations.

## Dataset Description

The dataset used is a Russian-language question-answering system, structured to support the task of detecting hallucinations in text. It consists of the following columns:

- **summary:** This column contains a summary or context related to the question and answer. It provides background information that may help in understanding whether the answer is relevant and correct.
- **question:** This column contains the question that is posed based on the summary.
- **answer:** This column contains the answer provided for the corresponding question.
- **is_hallucination:** This column is a binary label indicating whether the answer is a -hallucination (1) or not (0). A hallucination in this context refers to an incorrect or misleading answer that does not correctly respond to the question based on the provided summary.

### Dataset Size and Class Distribution

The dataset consists of a total of 1050 samples with the following class distribution:
- **Class 1 (Hallucinations)**: 532 samples
- **Class 0 (Non-Hallucinations)**: 518 samples

## Sample Data

Here are the first few rows of the dataset:


| line_id | summary                                                                                   | question                                           | answer                                       | is_hallucination |
|---------|-------------------------------------------------------------------------------------------|----------------------------------------------------|----------------------------------------------|------------------|
| 0       | Херманус Питер (Дик) Логгере (нидерл. Hermanus Pieter (Dick) Loggere, 6 мая 1921, Амстердам, Нидерланды — 30 декабря 2014, Хилверсюм, Нидерланды) — нидерландский хоккеист (хоккей на траве), полузащитник. Бронзовый призёр летних Олимпийских игр 1948 года. | В каком городе проходил чемпионат мира по хоккею с шайбой в 1936 году? | В Хилверсюме.                                | 1                |
| 1       | Ходуткинские горячие источники (Худутские горячие источники, Термальные источники вулкана Ходутка, Ходуткинское геотермальное месторождение) — пресные геотермальные источники на юге полуострова Камчатка в Елизовском районе Камчатского края. Относятся к Южно-Камчатской геотермальной провинции. | Как называется район в который входят источники?   | Елизовским районом                          | 0                |
| 2       | Чёрная вдова (лат. Latrodectus mactans) — вид пауков, распространённый в Северной и Южной Америке. Опасен для человека. | Для кого опасны пауки-бокоходы?                   | Для рыб.                                     | 1                |
| 3       | Рысь — река в России, протекает по территориям Муезерского городского поселения и Ледмозерского сельского поселения Муезерского района Карелии. Устье реки находится в 6,6 км по правому берегу реки Кайдодеги. Длина реки — 10 км. Рысь течёт преимущественно в северном направлении по заболоченной территории. Впадает в реку Кайдодеги возле озера Кайдодеги. Населённые пункты на реке отсутствуют. | Какова длина реки Рысь?                           | 5 км.                                        | 1                |
| 4       | И́се (яп. 伊勢市), ранее Удзиямада — город в Японии в префектуре Миэ. Исе входит в состав Национального парка Исе-Сима. Население — 98,819 человек (2003); город занимает площадь 178.97 км². Крупный центр туризма и паломничества.  Статус большого города Удзиямада получил 1 сентября 1906 года. После объединения с несколькими соседними городами 1 января 1955 года был образован город Исе. | Что такое Исе?                                  | Исе — это небольшой город в Японии, который не входит в состав Национального парка Исе-Сима. Население — 98,819 человек (2003); город занимает площадь 178.97 км². | 1                |


## Model

I use the 'DeepPavlov/rubert-base-cased' BERT model for sequence classification, pre-trained on a Russian language corpus, and fine-tune it on my dataset for the binary classification task of hallucination detection. This model is chosen for its proven effectiveness in understanding and generating Russian text.

## Tokenization

For tokenization, I use a compatible BertTokenizer. The tokenization process is as follows:

- Summary, question, and answer fields are concatenated into one string so that the model can get the context of all the information. Special tokens [CLS] and [SEP] are added:
    - The [CLS] token is added at the beginning of the entire sequence.
    - The [SEP] token is used to separate different segments of text, such as summary, question, and answer.
- token_type_ids is used to differentiate between different parts of the text (summary, question, and answer). Since the tokenizer from the transformers library does not support using more than two segments, summary and question are combined into one segment. Here, 0 indicates the first segment (summary and question), and 1 indicates the second segment (answer).
  
I do not lowercase text, as BERT is case-sensitive and was trained on case-sensitive text. Also, I do not remove special characters and punctuation, as they help the model to better understand context.

Maximum token length for the BERT model is 512 tokens. To handle large texts, I considered several methods and chose simple cropping to the maximum token length, as most examples in the dataset are within 512 tokens. Each segment (summary, question, and answer) of the text is cropped proportionally, so that the relative proportions of each part are maintained.

## Datasets

The dataset is randomly split into training and testing sets, with the test set size being 20%. A custom Dataset class and DataLoader are defined for the training and testing data, facilitating batch processing.

## Train Model

I train the model for 3 epochs to avoid overfitting. During each epoch, the training loss, accuracy, and F1-score are computed and printed. The model is also evaluated on the test set after each epoch. 

## Prediction and Evaluation

In the end, I make predictions on the training and test datasets and print a classification report and confusion matrix to evaluate the model's performance.

### Model Performance

#### Training Set Performance (Class 1):
 
- **Precision:** 0.98 
- **Recall:** 0.95 
- **F1-score:** 0.96 

#### Test Set Performance (Class 1):

- **Precision:** 0.93 
- **Recall:** 0.88 
- **F1-score:** 0.91 
 
The model demonstrates high performance on both training and test sets with only a slight decrease in metrics on the test set, which is expected for small dataset. However, the differences in performance are not significant, suggesting that the model can generalize well. 

## Content

- bert_crop.ipynb: Notebook with full code
- train.csv: Dataset
- requirements.txt: Required packages

## Requirements

ipython==8.22.2  
ipykernel==6.29.3  
torch==2.4.0  
numpy==1.26.4  
pandas==2.2.1  
tqdm==4.66.4  
scikit-learn==1.4.1.post1  
transformers==4.41.2  

## Create Virtual Environment

```bash
python3 -m venv new_env
source new_env/bin/activate
pip install -r requirements.txt

\# Adding a virtual environment to Jupyter as a new kernel
python -m ipykernel install --user --name=myenv --display-name "kernel name"
eactivate