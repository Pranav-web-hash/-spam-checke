# Group 2 Mini Projects

## Files
- project1_nlp_text_cleaner.py : NLP (run in Google Colab)
- project2_spam_classifier.py  : ML (Naive Bayes + TF-IDF)
- app.py + requirements.txt    : Cloud (Streamlit app)
- No dataset download needed: the code loads the SMS Spam Collection from a URL automatically

## Deploy Mini Project 3
1. Create a GitHub repo; upload app.py and requirements.txt
2. Go to share.streamlit.io, sign in with GitHub
3. New app -> choose repo -> main file: app.py -> Deploy
4. Copy the live link into your report

## IaaS vs PaaS vs SaaS (3 lines)
- IaaS: you rent raw servers/storage/network and manage the OS and apps yourself (e.g. AWS EC2).
- PaaS: provider manages servers and runtime; you only upload your code (e.g. Streamlit Cloud, Heroku).
- SaaS: a ready-made app used through a browser, nothing to manage (e.g. Gmail, Google Docs).

## Why Streamlit Cloud is PaaS
You only push code to GitHub; the platform handles servers, OS, Python runtime and hosting.

## Free tiers (verify on each official page, offers change)
| Provider | Free offer |
|----------|-----------|
| AWS | Free Tier: credits for new accounts plus always-free services such as Lambda (1M requests/month) |
| Azure | Free credit for first 30 days, 12 months of popular services free, plus always-free services |
| Google Cloud | $300 trial credit for 90 days, plus always-free products (e.g. e2-micro VM, Cloud Run limits) |
