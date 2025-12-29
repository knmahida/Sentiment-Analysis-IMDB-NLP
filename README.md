# Sentiment-Analysis-IMDB-NLP
End-to-end Sentiment Analysis project using NLP preprocessing, TF-IDF feature extraction, and Machine Learning models on IMDb movie reviews.

## Problem Statement
The objective of this project is to build a sentiment analysis model that classifies movie reviews from IMDb as positive or negative. This involves natural language processing (NLP) techniques to preprocess text data, extract features using TF-IDF, and train machine learning models to predict sentiment. The project demonstrates a complete pipeline from data loading to model evaluation, useful for understanding customer feedback, social media analysis, and automated review classification.

## Dataset Description
- **Source**: IMDb Dataset CSV file (`IMDB Dataset.csv`), containing movie reviews and their sentiments.
- **Features**:
  - review: Text of the movie review.
  - sentiment: Binary label ('positive' or 'negative').
- **Size**: 50,000 reviews (balanced: 25,000 positive, 25,000 negative).
- **Preprocessing**:
  - Text cleaning: Lowercasing, removing HTML tags, special characters, stopwords, and lemmatization.
  - Feature engineering: Added review length for EDA.
  - Encoding: Mapped sentiment to binary (1 for positive, 0 for negative).
- **Derived Features**: clean_review (preprocessed text), review_length.

## ML Approach
- **Task Type**: Binary classification (sentiment prediction).
- **Preprocessing**: NLTK for tokenization, stopword removal, and lemmatization.
- **Feature Extraction**: TF-IDF Vectorizer with max_features=5000 to convert text to numerical vectors.
- **Data Splitting**: 80% training, 20% testing (random_state=42).
- **Models**: Trained in pipelines combining TF-IDF and classifiers.
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix.

## Models Used
1. **Logistic Regression**: Linear model for binary classification.
2. **Naive Bayes (MultinomialNB)**: Probabilistic model suitable for text data.
3. **Support Vector Machine (LinearSVC)**: Effective for high-dimensional text features.

## Evaluation Metrics
- **Accuracy**: Overall correct predictions.
- **Precision, Recall, F1-Score**: Per-class metrics for positive and negative sentiments.
- **Confusion Matrix**: Visualized to show true positives, false positives, etc.

## Results
Based on typical runs:
- **Logistic Regression**: ~88-90% accuracy, balanced performance.
- **Naive Bayes**: ~85-87% accuracy, fast but slightly lower recall.
- **SVM**: Often the best performer, ~90-92% accuracy with high precision.
- SVM generally outperforms others due to its ability to handle sparse text features effectively.

## How to Run
1. **Prerequisites**: Python 3.8+, Jupyter Notebook.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset**: Place `IMDB Dataset.csv` in the project directory or update the path in the notebook.
4. **Run the Notebook**:
   ```bash
   jupyter notebook SentimentAnalysis_NLP.ipynb
   ```
5. Execute cells to preprocess data, train models, and view results.

## Key Learnings
- **Text Preprocessing**: Cleaning and lemmatization are crucial for reducing noise in NLP tasks.
- **Feature Extraction**: TF-IDF captures important word frequencies while ignoring common words.
- **Model Selection**: SVM often excels in text classification due to its margin-based optimization.
- **Pipeline Usage**: Scikit-learn pipelines simplify combining preprocessing and modeling steps.
- **Practical Insights**: Sentiment analysis can automate feedback analysis, aiding businesses in understanding customer opinions.
