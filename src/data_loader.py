import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # zet de tekst om in getallen zodat ML het begrijpt

def load_and_prepare_data():
    # Pad naar de dataset
    data_path = "data/IMDB Dataset.csv"

    # Dataset inladen
    df = pd.read_csv(data_path)

    # Kolommen hernoemen
    df.columns = ["review", "sentiment"]

    # Alleen 2000 willekeurige reviews gebruiken
    df_sample = df.sample(n=2000, random_state=42)

    # Sentiment omzetten naar numerieke waarden: positief = 1, negatief = 0
    df_sample["sentiment"] = df_sample["sentiment"].map({"positive": 1, "negative": 0})

    # Train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample["review"], df_sample["sentiment"], test_size=0.2, random_state=42
    )

    # Tekst vectoriseren
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer