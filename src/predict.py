import joblib
import os

def predict_text(model_path, vectorizer, texts):
    if not os.path.exists(model_path):
        raise FileExistsError(f"Modelbestand niet gevonden: {model_path}")

    model = joblib.load(model_path)

    X_vect = vectorizer.transform(texts)

    predictions = model.predict(X_vect)

    return predictions