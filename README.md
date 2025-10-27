# Movie Review Sentiment Classifier

## Quickly analyze whether a movie review is positive or negative!

This application is a small machine learning project that classifies movie reviews as **positive** or **negative**. It is built using Python and scikit-learn, allowing you to train, evaluate, and predict the sentiment of new reviews quickly.

## Table of Contents

- [Advantages](#advantages)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)

## Advantages
### Why you should choose this application:
* Quickly determine the sentiment of movie reviews
* Lightweight Python project using standard ML libraries
* Structured code with modular design (data loader, training, prediction)
* Can be extended with other models or datasets
* Easy to train on new reviews without advanced setup

## Installation
1. Clone this project:
2. Ensure you have Python 3.13 installed (or compatible version)
3. Make sure the dataset IMDB Dataset.csv is located in src/data/
4. Open the project in your preferred IDE (e.g., PyCharm)
5. Run the application:
   - python src/main.py

If you encounter any issues during installation, feel free to contact me via:
📧 sabritafersit1@gmail.com

## Usage
When you run the application, it will:
- Load and preprocess the dataset
- Train and evaluate machine learning models
- Predict sentiment for new movie reviews

Add your reviews to the `new_reviews` list in `main.py` and run the script:
- I absolutely loved this movie! Amazing story and acting.
- Worst movie ever. I hated it so much.

The output will show whether each review is Positive or Negative.

## Roadmap

Based on a feedback session I held with my buddy, I want to add these improvements in the future:
- Implement a simple interface that allows users to input reviews without modifying the code: right now the user has to modify the code to input a review, it would better if there was a small interface that allows the user to input reviews without having to edit the code.
- Adding a neutral predicition category: right now the review can be classified as "Positive" and "Negative", in the future I would like to add a "Neutral" sentiment category.
- Using a larger dataset for higher accuracy

The most important improvements are prioritized at the top, while smaller enhancements are planned for later iterations.
