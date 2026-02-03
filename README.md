# IMDB Sentiment Analysis

This repository contains a comprehensive implementation of a sentiment analysis model designed to classify IMDB movie reviews as either positive or negative. Leveraging Natural Language Processing (NLP) techniques and Logistic Regression, this project demonstrates a robust pipeline for text classification tasks.

## Abstract

Sentiment analysis, a subfield of Natural Language Processing, involves determining the emotional tone behind a body of text. This project utilizes the IMDB dataset containing 50,000 labeled reviews to train a Logistic Regression model. The solution incorporates text preprocessing, TF-IDF vectorization, and model evaluation to achieve high-accuracy predictions.

## Dataset

The model is trained on the **IMDB Dataset of 50K Movie Reviews**.
- **Source**: [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 records
- **Labels**: Balanced classes (Positive / Negative)

## Methodology

The development pipeline follows these standard machine learning stages:

1.  **Data Preprocessing**:
    *   HTML tag removal.
    *   Tokenization and normalization.
    *   Stopword removal using NLTK.
    *   Noise reduction (punctuation and special character removal).

2.  **Feature Extraction**:
    *   Implementation of **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to transform textual data into numerical feature vectors.

3.  **Model Architecture**:
    *   **Algorithm**: Logistic Regression.
    *   **Rationale**: Chosen for its efficiency and interpretability in binary classification tasks involving sparse high-dimensional data.

4.  **Evaluation**:
    *   Metrics: Accuracy, Precision, Recall, and F1-Score.
    *   Visualization: Confusion Matrix for error analysis.

## Prerequisites

Ensure the following dependencies are installed in your Python environment:

*   Python 3.x
*   Pandas
*   NumPy
*   NLTK
*   Scikit-learn
*   Matplotlib
*   Seaborn
*   ipywidgets

## Installation and Usage

To replicate this analysis locally:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/UmmeKulsumTumpa/Sentiment-Analysis-IMDB-Movie-Reviews.git
    cd IMDB-Sentiment-Analysis
    ```

2.  **Execute the Analysis**:
    *   Open `Sentiment_Analysis_IMDB.ipynb` in Jupyter Notebook or JupyterLab.
    *   Execute the cells sequentially to preprocess data, train the model, and view results.

3.  **Interactive Inference**:
    *   The notebook includes an interactive widget allowing for real-time inference on custom user input.

## Performance Results

The Logistic Regression model achieved the following performance metrics on the test set:

*   **Accuracy**: ~88%
*   **Generalization**: The model demonstrates consistent performance across both classes with balanced precision and recall scores.

## Project Structure

```text
IMDB-Sentiment-Analysis/
├── Sentiment_Analysis_IMDB.ipynb   # Analysis and Modeling Notebook
├── IMDB Dataset.csv                # Training Data
└── README.md                       # Project Documentation
```

## Future Scope

Potential areas for optimization and extension include:
*   **Deep Learning Integration**: Implementation of LSTM (Long Short-Term Memory) networks or Transformer-based architectures (e.g., BERT) for enhanced context understanding.
*   **Deployment**: Packaging the model as a REST API using Flask or FastAPI.
*   **Hyperparameter Tuning**: employing Grid Search or Randomized Search to further optimize the Logistic Regression parameters.

