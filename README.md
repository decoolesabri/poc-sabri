# 🎬 Sentiment Analysis on Movie Reviews

A simple Proof of Concept (PoC) that analyzes movie reviews and predicts whether a review is **positive** or **negative** using machine learning.

## Project Overview
This project uses the **IMDB Dataset** and applies text preprocessing, TF-IDF vectorization, and two models — **Naive Bayes** and **Logistic Regression** — to classify movie reviews based on sentiment.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/movie-sentiment-poc.git
   ```
2. Navigate into the project folder
   ```bash
   cd poc-sabri
   ```
3. Prepare the dataset
   - Go to the folder `dataset/`
   - Copy the file `IMDB Dataset.csv`
   - Paste it into `src/data/`

   The project will not run without this file in `src/data/`.

4. Install dependencies
   ```bash
   pip install pandas scikit-learn joblib
   ```
5. Run the project
   ```bash
   python src/main.py
   ```

## Usage
Navigate to the main.py file and edit the "new_reviews" list and add the review you want to test:
  ```bash
  new_reviews = [
      "(write your review here)"
  ]
  ```

## Roadmap

Based on a feedback session I held with my buddy, I want to add these improvements in the future:

- **Frontend interface** – Add a simple web interface (e.g., using Streamlit or Flask) where users can enter a review and instantly see the predicted sentiment.
- **Adding a neutral predicition category** - right now the review can be classified as "Positive" and "Negative", in the future I would like to add a "Neutral" sentiment category.
- **Data expansion** – Train the model on a larger and more diverse dataset of movie reviews to improve generalization.
- **Language support** – Add multilingual sentiment analysis (e.g., English and Dutch).

These features are not required for the current proof of concept but would make the project more complete and user-friendly in the future. The most important improvements are prioritized at the top.
