# 📝 Project Handwritten Reference Guide (Easy & Short)

**Project Title:** Email Intent Classification with Attention-Based Sequence Models and Priority Prediction

---

## 📌 1. Project Title & Dataset Overview

### 🏷️ Project Title
**Email Intent Classification with Attention-Based Sequence Models and Priority Prediction**

### 📊 Dataset Details
- **Dataset Name**: `b4b4yg4/email-intent`
- **Source**: Hugging Face Datasets
- **Total Emails**: 2,533 customer emails
- **Intent Classes**: 17 categories (e.g., `delivery_issue`, `refund_request`, `technical_support`, `invoice_request`)
- **Priority Tiers**: 3 levels (`High`, `Medium`, `Low`)
- **Average Email Length**: ~13 words (~77 characters)

---

## 🧹 2. Text Preprocessing Techniques (Short Theory)

### 1. Missing Value Handling
- **Definition**: Checking and removing null/empty text or label entries.
- **Why Use It**: Machine learning algorithms crash if input data contains missing values (`NaN`).

### 2. Text Lowercasing
- **Definition**: Converting all uppercase text to lowercase (e.g., `"URGENT"` ➔ `"urgent"`).
- **Why Use It**: Treats `"Order"`, `"ORDER"`, and `"order"` as the exact same word.

### 3. HTML & URL Removal
- **Definition**: Removing website links (`http://...`) and web tags (`<p>`, `<div>`).
- **Why Use It**: Web links and tags add noise without providing intent meaning.

### 4. Special Character Cleaning
- **Definition**: Removing symbols (`@`, `#`, `$`, `%`, `*`) while keeping letters and numbers.
- **Why Use It**: Reduces vocabulary size and cleans noisy text.

### 5. Whitespace Normalization
- **Definition**: Replacing multiple spaces, tabs, and newlines with a single space.
- **Why Use It**: Ensures clean, uniform text layout.

### 6. Tokenization
- **Definition**: Converting raw text strings into numerical word IDs (tokens).
- **Why Use It**: Neural networks cannot read English words directly; they only process numbers.

### 7. Sequence Padding & Truncating
- **Definition**: Equalizing all email sequence lengths to a fixed size (`max_len = 30`) using zero-padding.
- **Why Use It**: Deep learning models require fixed-size matrix inputs.

### 8. Trainable Word Embeddings
- **Definition**: Converting integer token IDs into dense continuous vector representations (e.g., 64 dimensions).
- **Why Use It**: Captures semantic relationships (words with similar meanings have similar vector values).

---

## 🤖 3. Machine Learning & Deep Learning Algorithms (Short Theory)

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
- **Definition**: A feature extraction technique that calculates word importance.
  - **Term Frequency (TF)**: How frequently a word appears in an email.
  - **Inverse Document Frequency (IDF)**: Penalizes common words (`"the"`, `"is"`) and boosts rare important words (`"refund"`, `"damaged"`).
- **Formula**: $\text{TF-IDF} = \text{TF} \times \log\left(\frac{N}{\text{DF}}\right)$

### 2. Logistic Regression (Baseline Model)
- **Definition**: A classical linear classification model.
- **How It Works**: Computes a linear weighted sum of TF-IDF features and applies Softmax to calculate class probabilities.
- **Why Used**: Fast, simple, and highly accurate baseline for text classification.

### 3. LSTM (Long Short-Term Memory)
- **Definition**: A specialized Recurrent Neural Network (RNN) designed for sequential text data.
- **How It Works**: Uses memory gates (**Input Gate**, **Forget Gate**, **Output Gate**) to pass context through time steps without losing long-term memory.
- **Why Used**: Captures word order and sentence context better than simple Bag-of-Words.

### 4. Attention Mechanism (Main Proposed Model)
- **Definition**: A layer that assigns dynamic importance weights ($\alpha_t$) to every word in a sequence.
- **How It Works**: Calculates how important each word is for the final prediction and sums them into a weighted **Context Vector**.
- **Why Used**: Standard LSTMs squeeze the entire sentence into one final vector, losing details. Attention allows the model to "focus" on key words like *"urgent"*, *"failed"*, or *"refund"*.
- **Formula**: $\text{Context Vector} = \sum_{t} \alpha_t \cdot h_t$

### 5. Softmax Activation Function
- **Definition**: An activation function placed at the output layer of a multi-class neural network.
- **How It Works**: Converts raw output scores (logits) into probabilities that sum to 1.0 (100%).

### 6. Categorical Cross-Entropy Loss
- **Definition**: The loss function used during neural network training.
- **How It Works**: Measures the error between predicted class probabilities and true target labels.

---

## 📊 4. Model Performance Comparison

| Model | Technique / Architecture | Accuracy | F1-Score | Key Insight |
| :--- | :--- | :---: | :---: | :--- |
| **Model 1 (Baseline)** | TF-IDF + Logistic Regression | **90.00%** | **0.8987** | Fast & highly accurate on short keyword-rich emails. |
| **Model 2 (Basic DL)** | Embedding + LSTM + Dense | 43.68% | 0.4122 | Suffers from information bottlenecking in standard LSTM. |
| **Model 3 (Proposed Model)** | Embedding + LSTM + **Attention** | **68.42%** | **0.6818** | **+24.7% accuracy boost** over basic LSTM due to attention weights. |
| **Priority Model** | Embedding + LSTM + Attention + Priority Head | **54.67%** | **0.4769** | Successfully classifies `High`, `Medium`, `Low` urgency tiers. |

---

## 💡 Quick Viva Questions & Answers (Bonus)

- **Q1: Why did Logistic Regression get higher accuracy than LSTM?**  
  *Answer:* The emails in this dataset are short (~13 words) and contain strong keyword triggers (e.g. "invoice", "refund"). Linear models on TF-IDF n-grams capture sparse keyword rules directly without needing large training data.

- **Q2: What is the main contribution of your proposed model?**  
  *Answer:* Adding a **Simple Attention Layer** to the LSTM improved sequence accuracy from **43.68% to 68.42%** (+24.7% gain), demonstrating that attention helps the model focus on key words.
