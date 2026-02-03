# Sentiment Analysis of IMDB Movie Reviews

## 📌 Overview
This repository hosts a machine learning project dedicated to performing sentiment analysis on the IMDB movie reviews dataset. By leveraging **Natural Language Processing (NLP)** techniques and a **Logistic Regression** classifier, the model accurately categorizes reviews as either **positive** or **negative**.

The project demonstrates the end-to-end pipeline of a text classification task, including data preprocessing, feature extraction using TF-IDF, model training, and performance evaluation.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technologies](#-technologies)
- [Dataset](#-dataset)
- [Installation & Usage](#-installation--usage)
- [Model Performance](#-model-performance)
- [Visualizations](#-visualizations)
- [Future Enhancements](#-future-enhancements)

---

## 🔑 Key Features
*   **Robust Text Preprocessing**: Implementation of tokenization, and removal of punctuation and stopwords to clean raw text data.
*   **Feature Engineering**: Utilization of **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to transform text into meaningful numerical representations.
*   **Predictive Modeling**: Training of a Logistic Regression model optimized for binary classification.
*   **Interactive Interface**: A user-friendly widget powered by `ipywidgets` allowing for real-time sentiment prediction on custom user input.

---

## 🛠 Technologies
The project is built using the **Python** ecosystem with the following core libraries:
*   **Data Manipulation**: `Pandas`, `NumPy`
*   **Natural Language Processing**: `NLTK`
*   **Machine Learning**: `Scikit-learn`
*   **Visualization**: `Matplotlib`, `Seaborn`
*   **Interface**: `ipywidgets`

---

## 📂 Dataset
The project utilizes the **IMDB Dataset**, a widely recognized benchmark for binary sentiment classification, containing:
*   **50,000** movie reviews.
*   Balanced classes (25k positive, 25k negative).

---

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/mounimRhouli/IMDB-Sentiment-Analysis.git
    cd IMDB-Sentiment-Analysis
    ```

2.  **Environment Setup**
    Ensure you have Python installed. It is recommended to use a virtual environment. Install dependencies via pip or conda if a `requirements.txt` is present (or manually install the libraries listed above).

3.  **Execution**
    *   Launch Jupyter Notebook:
        ```bash
        jupyter notebook Sentiment_Analysis_IMDB.ipynb
        ```
    *   Execute the cells sequentially to preprocess data, train the model, and evaluate results.
    *   Use the interactive UI at the end of the notebook to test custom reviews.

---

## 📊 Model Performance
The classification model achieves competitive performance metrics on the test set:
*   **Accuracy**: **88%**
*   **Generalization**: The model demonstrates balanced Precision and Recall scores across both classes, indicating robust handling of both positive and negative sentiments.

---

## 📈 Visualizations

### Confusion Matrix
Evaluation of the model's true positive and negative predictions.

![Confusion Matrix](assets/confusion-matrix.png)

### Interactive UI
Screenshot of the real-time classification interface.

![UI Screenshot](assets/ui.png)

---

## ✨ Future Enhancements
*   **Deep Learning Integration**: Experimentation with architectures such as **LSTMs** or Transformer-based models (**BERT**) to capture deeper contextual nuances.
*   **Web Deployment**: Development of a standalone web application using **Flask** or **Streamlit** for broader accessibility.
*   **Multilingual Support**: Extending the preprocessing pipeline to support sentiment analysis for non-English reviews.

