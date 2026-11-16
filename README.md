---
title: Bank Deposit Prediction
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🏦 Bank Term Deposit Prediction

Gradio app that predicts whether a bank customer will subscribe to a term deposit, using a
StackingClassifier (SVC + ExtraTrees + RandomForest + XGBoost + LightGBM, with a Logistic Regression
meta-model) trained on a bank marketing campaign dataset.

⚠️ The `duration` feature (last call length) is only known after the call, so predictions are
meant for post-call analysis, not for forecasting before contacting the customer.
For learning/demo purposes only.
