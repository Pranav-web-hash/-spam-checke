# Mini Project 2: Spam Classifier (Machine Learning)
# Dataset: Kaggle "SMS Spam Collection" -> spam.csv
# Colab: !pip install scikit-learn pandas matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Load data
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. TF-IDF
tfidf = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)

# 4. Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_t, y_train)

# 5. Metrics
pred = model.predict(X_test_t)
print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 6. Test with 3 messages of your own (change these)
my_msgs = [
    "Congratulations! You won a free iPhone. Click the link to claim now",
    "Hey, are we meeting for the lab practical tomorrow?",
    "URGENT: Your bank account is blocked. Send OTP to verify",
]
for msg, p in zip(my_msgs, model.predict(tfidf.transform(my_msgs))):
    print(f"{'SPAM' if p else 'HAM '} -> {msg}")
