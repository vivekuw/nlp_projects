import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Email Intent Classification & Priority Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Attention Layer
@tf.keras.utils.register_keras_serializable(package="Custom")
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'subj:\s*case\s*\d+\s*-\s*', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_assets():
    models_dir = "models"
    tokenizer = joblib.load(os.path.join(models_dir, "tokenizer.pkl"))
    intent_le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    priority_le = joblib.load(os.path.join(models_dir, "priority_label_encoder.pkl"))
    
    intent_model = load_model(
        os.path.join(models_dir, "attention_model.keras"),
        custom_objects={'SimpleAttention': SimpleAttention}
    )
    priority_model = load_model(
        os.path.join(models_dir, "priority_model.keras"),
        custom_objects={'SimpleAttention': SimpleAttention}
    )
    return tokenizer, intent_le, priority_le, intent_model, priority_model

def predict_email(text, tokenizer, intent_le, priority_le, intent_model, priority_model, max_len=30):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    intent_probs = intent_model.predict(padded, verbose=0)[0]
    intent_idx = np.argmax(intent_probs)
    intent_label = intent_le.inverse_transform([intent_idx])[0]
    intent_conf = float(intent_probs[intent_idx]) * 100
    
    priority_probs = priority_model.predict(padded, verbose=0)[0]
    priority_idx = np.argmax(priority_probs)
    priority_label = priority_le.inverse_transform([priority_idx])[0]
    priority_conf = float(priority_probs[priority_idx]) * 100
    
    return {
        'cleaned_text': cleaned,
        'intent': intent_label,
        'intent_confidence': intent_conf,
        'priority': priority_label,
        'priority_confidence': priority_conf
    }

# Main Application
def main():
    st.title("📧 Email Intent Classification with Attention-Based Sequence Models & Priority Prediction")
    st.markdown("### Beginner-Friendly End-to-End NLP & Deep Learning Analytics Dashboard")
    st.markdown("---")

    # Navigation Sidebar
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio(
        "Select Section:",
        [
            "Project Overview",
            "Dataset Overview & EDA",
            "Word Cloud",
            "Model Performance & Comparison",
            "Live Email Prediction System",
            "Conclusion & Key Findings"
        ]
    )

    # Load data & metrics safely
    cleaned_df = pd.read_csv(os.path.join("data", "processed", "cleaned_emails.csv"))
    comp_df = pd.read_csv(os.path.join("results", "metrics", "final_model_comparison.csv"))

    if page == "Project Overview":
        st.header("🎯 Project Overview & Objective")
        st.write("""
        This project builds an end-to-end NLP data analysis pipeline designed to automatically classify customer support email intents and predict their urgency/priority level.
        
        It compares traditional machine learning baseline models against deep sequence architectures (LSTM) and an **Attention-Based LSTM** proposed model.
        """)

        st.subheader("📌 Key Features:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🤖 **Models Evaluated**\n- TF-IDF + Logistic Regression\n- Basic LSTM\n- LSTM + Attention (Proposed)")
        with col2:
            st.success("🎯 **Dual Output Prediction**\n- 17 Intent Categories\n- 3 Priority Levels (High, Medium, Low)")
        with col3:
            st.warning("⚡ **Explainable Architecture**\n- Custom Keras Attention Mechanism\n- End-to-End Preprocessing")

        st.subheader("🔗 Dataset Source & Credits")
        st.markdown("- **Dataset**: `b4b4yg4/email-intent`")
        st.markdown("- **Source**: [Hugging Face Email Intent Dataset](https://huggingface.co/datasets/b4b4yg4/email-intent)")

    elif page == "Dataset Overview & EDA":
        st.header("📊 Dataset Overview & Statistics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Emails", f"{len(cleaned_df):,}")
        col2.metric("Intent Classes", f"{cleaned_df['intent'].nunique()}")
        col3.metric("Annotated Priority Labels", f"{cleaned_df['urgency'].dropna().count():,}")
        col4.metric("Average Word Count", f"{cleaned_df['word_count'].mean():.1f} words")

        st.markdown("---")
        st.subheader("📈 Key Visualizations")

        tab1, tab2, tab3 = st.tabs(["Intent Distribution", "Priority Distribution", "Email Length Distribution"])

        with tab1:
            st.image(os.path.join("results", "figures", "intent_distribution.png"), caption="Distribution of 17 Email Intent Classes", use_container_width=True)

        with tab2:
            st.image(os.path.join("results", "figures", "priority_distribution.png"), caption="Distribution of Priority/Urgency Annotations", use_container_width=True)

        with tab3:
            st.image(os.path.join("results", "figures", "email_length_distribution.png"), caption="Character and Word Length Histograms", use_container_width=True)

    elif page == "Word Cloud":
        st.header("☁️ Email Corpus Word Cloud")
        st.write("Visualizing the most frequent vocabulary terms across cleaned customer email messages.")
        st.image(os.path.join("results", "figures", "wordcloud.png"), caption="Word Cloud of Cleaned Email Texts", use_container_width=True)

    elif page == "Model Performance & Comparison":
        st.header("🏆 Model Performance Comparison")
        
        st.subheader("📊 Comparative Metrics Table")
        st.dataframe(
            comp_df[['Model', 'Architecture', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1_Score': '{:.4f}'
            }),
            use_container_width=True
        )

        st.subheader("📈 Visual Comparison & Metrics")
        st.image(os.path.join("results", "figures", "model_comparison.png"), caption="Accuracy, Precision, Recall, and F1-Score Bar Chart Comparison", use_container_width=True)

        st.subheader("🌀 Confusion Matrices")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(os.path.join("results", "figures", "logistic_regression_confusion_matrix.png"), caption="Logistic Regression Confusion Matrix", use_container_width=True)
        with c2:
            st.image(os.path.join("results", "figures", "lstm_confusion_matrix.png"), caption="Basic LSTM Confusion Matrix", use_container_width=True)
        with c3:
            st.image(os.path.join("results", "figures", "attention_confusion_matrix.png"), caption="Proposed LSTM + Attention Confusion Matrix", use_container_width=True)

        st.subheader("📉 Multi-Class ROC Curves")
        st.image(os.path.join("results", "figures", "roc_curve.png"), caption="ROC Curves for Proposed Attention Model", use_container_width=True)

    elif page == "Live Email Prediction System":
        st.header("🔮 Live Email Intent & Priority Predictor")
        st.write("Enter a sample email message below to test live classification with trained models:")

        try:
            tokenizer, intent_le, priority_le, intent_model, priority_model = load_assets()

            default_text = "Subj: Case 0099 - Order 51099 laptop never arrived, tracking is stuck! Need urgent refund."
            user_input = st.text_area("Customer Email Text:", value=default_text, height=120)

            if st.button("🚀 Classify Email", type="primary"):
                if not user_input.strip():
                    st.warning("Please enter a valid email message.")
                else:
                    res = predict_email(user_input, tokenizer, intent_le, priority_le, intent_model, priority_model)
                    
                    st.markdown("---")
                    st.subheader("🎯 Prediction Output")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Intent", value=res['intent'])
                        st.write(f"**Intent Confidence:** {res['intent_confidence']:.2f}%")
                        st.progress(res['intent_confidence'] / 100.0)

                    with col2:
                        priority_color = "🔴" if res['priority'] == 'High' else ("🟡" if res['priority'] == 'Medium' else "🟢")
                        st.metric(label="Predicted Priority", value=f"{priority_color} {res['priority']}")
                        st.write(f"**Priority Confidence:** {res['priority_confidence']:.2f}%")
                        st.progress(res['priority_confidence'] / 100.0)

                    st.info(f"**Cleaned Email Input:** `{res['cleaned_text']}`")

        except Exception as e:
            st.error(f"Error loading models or generating prediction: {e}")

    elif page == "Conclusion & Key Findings":
        st.header("📝 Conclusion & Key Findings")
        st.markdown("""
        ### Key Insights:
        1. **Baseline Strong Performance**: TF-IDF + Logistic Regression achieved **89.87% F1-score** due to distinct keyword patterns in short customer service queries.
        2. **Sequence Modeling & Attention Advantage**: The **LSTM + Attention model (74.45% F1-score)** significantly outperformed the **Basic LSTM (54.73% F1-score)**, demonstrating how dynamic attention weights help sequence models focus on crucial keywords like *"urgent"*, *"refund"*, or *"damaged"*.
        3. **Dual Priority Classification**: Priority prediction successfully mapped customer emails into `High`, `Medium`, and `Low` urgency tiers.
        
        ### 🚀 Applications & Future Scope:
        - **Automated Support Helpdesks**: Prioritizing urgent billing and delivery tickets.
        - **Multilingual Support**: Extending tokenization to handle international queries.
        - **Transformer Exploration**: Comparing sequence attention against lightweight Transformer encodings in future iterations.
        """)

if __name__ == "__main__":
    main()
