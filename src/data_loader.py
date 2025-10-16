from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_prepare_data():
    # Ik gebruik hier een kleine dataset als tijdelijk voorbeeld
    categories = ['rec.autos', 'rec.sport.hockey']
    data  = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Tekst vectoriseren
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer