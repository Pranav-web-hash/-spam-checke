# Mini Project 3: one-file Streamlit spam checker
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

st.set_page_config(page_title="Spam Checker", page_icon="📩")


@st.cache_resource
def load_model():
    df = pd.read_csv(URL, sep="\t", header=None, names=["label", "text"])
    y = (df["label"] == "spam").astype(int)
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["text"])
    return tfidf, MultinomialNB().fit(X, y)


tfidf, model = load_model()

st.title("📩 SMS Spam Checker")
st.write("Type a message and the Naive Bayes model will tell you if it is spam.")

msg = st.text_area("Your message")
if st.button("Check"):
    if msg.strip():
        vec = tfidf.transform([msg])
        prob = model.predict_proba(vec)[0][1]
        if prob >= 0.5:
            st.error(f"SPAM (confidence {prob:.0%})")
        else:
            st.success(f"Not spam (spam probability {prob:.0%})")
    else:
        st.warning("Please type a message first.")
