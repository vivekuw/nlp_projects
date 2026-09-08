# Email Intent Classification with Attention-Based Sequence Models and Priority Prediction

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)

An end-to-end beginner-friendly NLP data analysis and machine learning project that classifies customer email intents and predicts email priority/urgency using traditional machine learning baseline models, deep learning sequence models (LSTM), and a **Proposed Attention-Based Sequence Model**.

---

## 🎯 Project Objectives
1. **Multi-Class Intent Classification**: Automatically categorize customer support emails into 17 intent categories.
2. **Priority Triage**: Predict urgency levels (`High`, `Medium`, `Low`) for annotated email queries.
3. **Model Comparison**: Compare TF-IDF + Logistic Regression against Basic LSTM and Proposed LSTM + Attention models.
4. **Interactive Analytical Dashboard**: Provide a Streamlit web application for real-time visualization and live email prediction.

---

## 📊 Dataset Information
- **Dataset Name**: `b4b4yg4/email-intent`
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/b4b4yg4/email-intent)
- **Total Records**: **2,533 emails** (Train split: 2,433, Test split: 100)
- **Columns**: `email_text`, `intent`, `urgency`, `order_id`, `product`
- **Intent Classes**: 17 categories
- **Urgency Annotations**: 1,000 annotated samples (`High`: 403, `Medium`: 389, `Low`: 208)

---

## 🏆 Final Model Performance Comparison

| Model | Feature / Architecture | Accuracy | Precision | Recall | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | TF-IDF + Logistic Regression | **0.9000** | **0.9017** | **0.9000** | **0.8987** |
| **Basic LSTM** | Embedding + LSTM + Dense | 0.4368 | 0.4343 | 0.4368 | 0.4122 |
| **Proposed Model (LSTM + Attention)** | Embedding + LSTM + Attention + Dense | **0.6842** | **0.7025** | **0.6842** | **0.6818** |
| **Priority Classifier** | Embedding + LSTM + Attention + Priority Dense | **0.5467** | **0.4510** | **0.5467** | **0.4769** |

---

## 📁 Project Directory Structure
```
email-intent-classification/
│
├── data/
│   ├── raw/
│   │   ├── email_intent_full_raw.csv
│   │   ├── email_intent_train_raw.csv
│   │   └── email_intent_test_raw.csv
│   └── processed/
│       ├── cleaned_emails.csv
│       ├── train_emails.csv
│       ├── val_emails.csv
│       └── test_emails.csv
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing_tfidf.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_lstm_model.ipynb
│   ├── 05_attention_model.ipynb
│   ├── 06_priority_prediction.ipynb
│   └── 07_final_evaluation.ipynb
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── priority_label_encoder.pkl
│   ├── logistic_regression_model.pkl
│   ├── lstm_model.keras
│   ├── attention_model.keras
│   └── priority_model.keras
│
├── results/
│   ├── figures/
│   │   ├── intent_distribution.png
│   │   ├── priority_distribution.png
│   │   ├── email_length_distribution.png
│   │   ├── wordcloud.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── lstm_training_history.png
│   │   ├── lstm_confusion_matrix.png
│   │   ├── attention_training_history.png
│   │   ├── attention_confusion_matrix.png
│   │   ├── priority_confusion_matrix.png
│   │   ├── model_comparison.png
│   │   └── roc_curve.png
│   ├── metrics/
│   │   ├── logistic_regression_metrics.csv
│   │   ├── lstm_metrics.csv
│   │   ├── attention_metrics.csv
│   │   ├── priority_metrics.csv
│   │   └── final_model_comparison.csv
│   └── project_results.md
│
├── app.py             # Streamlit Interactive Analytical Dashboard
├── requirements.txt   # Python Dependencies
├── README.md          # Project Overview & Guide
└── .gitignore         # Git Exclusions
```

---

## 💻 How to Run the Project

### 1. Change Directory to Project Folder:
```bash
cd p:\nlp_projects\email-intent-classification
```

### 2. Install Required Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Analytical Dashboard:
```bash
streamlit run app.py
```

---

## ⚡ Key Highlights for Viva Presentation
1. **Explainable Text Preprocessing**: Lowercasing, HTML/URL filtering, non-aggressive special character removal preserving sentiment/urgency keywords (`not`, `failed`, `urgent`).
2. **Data Leakage Avoidance**: Fitting TF-IDF Vectorizer and Tokenizer ONLY on training set (`X_train`), transforming validation and testing sets.
3. **Custom Attention Layer**: Demonstrating how dynamic attention weights $\alpha_t$ improve standard LSTM sequence performance by **+24.7% accuracy**.
4. **Live Interactive Dashboard**: Testing custom customer email inputs in real-time with confidence score progress bars.
