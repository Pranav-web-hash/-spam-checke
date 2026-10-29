# Mini Project 1: Text Cleaner and Keyword Finder (NLP)
# Dataset: Kaggle "SMS Spam Collection" -> spam.csv (columns v1 = label, v2 = text)
# Colab: upload spam.csv, then run:  !pip install nltk scikit-learn matplotlib pandas
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")

# 1. Load 200 short texts
df = pd.read_csv("spam.csv", encoding="latin-1")[["v2"]].rename(columns={"v2": "text"})
df = df.head(200)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# 2. Clean: lowercase -> remove punctuation -> remove stop words -> stem
def clean(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)  # drop numbers (optional)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(stemmer.stem(w) for w in words)


df["clean"] = df["text"].apply(clean)
print(df[["text", "clean"]].head())

# 3. 10 most common words -> bar chart
counts = Counter(" ".join(df["clean"]).split()).most_common(10)
words, freq = zip(*counts)
plt.figure(figsize=(8, 4))
plt.bar(words, freq, color="steelblue")
plt.title("Top 10 most common words")
plt.xlabel("Word (stemmed)")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_words.png")
plt.show()

# 4. Top 10 TF-IDF keywords
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(df["clean"])
scores = matrix.sum(axis=0).A1
top = sorted(zip(tfidf.get_feature_names_out(), scores), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 TF-IDF keywords:")
for word, score in top:
    print(f"{word:15s} {score:.3f}")
