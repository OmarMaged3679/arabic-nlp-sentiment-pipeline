# 📝 Arabic Sentiment Analysis with Word2Vec

This project performs sentiment classification on Arabic text using pre-trained word embeddings from **AraVec** and classic machine learning models. It classifies each sentence into one of three categories:

- **Positive**
- **Negative**
- **Mixed**

---

## 📂 Dataset

The dataset used for training is included in this repository:  
➡️ [`Dataset.tsv`](./Dataset.tsv)

It contains Arabic text samples with corresponding sentiment labels.

---

## 🔧 What the Notebook Does

- Loads and explores the dataset
- Cleans and tokenizes Arabic text
- Converts sentences into vectors using pre-trained **AraVec Word2Vec** embeddings
- Trains and evaluates two models:
  - **SGDClassifier**
  - **XGBoost**
- Visualizes confusion matrices
- Tests the model on real-world Arabic examples

---

## 🚀 How to Run

This project is designed to run in **Google Colab**.

In my setup, I loaded the dataset from my **Google Drive** using:

```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Dataset.tsv', sep='\t')
````

You can either do the same, or modify that line to load the dataset locally (e.g., `pd.read_csv('Dataset.tsv', sep='\t')`).

### To follow the same setup:

1. **Make a copy of the dataset**
   Upload `Dataset.tsv` to your own **Google Drive** under:
   `MyDrive/Dataset.tsv`

2. Open the notebook in Colab using the button below

3. Run all cells — everything else (AraVec download, setup, training) is handled in the notebook

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OmarMaged3679/arabic-nlp-sentiment-pipeline/blob/main/arabic_sentiment_analysis.ipynb)

---

## 📌 Example Output

Sample prediction on new text:
```python
["لم يعجبني هذا المنتج، الجودة ضعيفة.",
 "هذا المنتج رائع وجودته ممتازة.",
 "بالرغم من السعر المنخفض، الجودة ممتازة ولكن التوصيل تأخر كثيرًا."]
````

Output:

```
['Negative', 'Positive', 'Mixed']
```

---

## 🤖 Tools & Libraries

* Python, Pandas, NumPy
* NLTK
* Gensim (AraVec embeddings)
* Scikit-learn
* XGBoost
* Matplotlib

---

## 📜 License

For educational use. AraVec embeddings are credited to their [original authors](https://github.com/bakrianoo/aravec).
