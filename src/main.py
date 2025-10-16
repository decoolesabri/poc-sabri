from data_loader import load_and_prepare_data

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
    print("Klaar! Dataset succesvol geladen 🎉")
    print("Aantal trainingsvoorbeelden:", X_train.shape[0])
    print("Aantal testvoorbeelden:", X_test.shape[0])