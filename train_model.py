import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. Load Processed Dataset
# ==========================================

df = pd.read_csv("processed_job_description_dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))
print()


# ==========================================
# 2. Prepare Input and Target
# ==========================================

X_text = df["clean_description"].fillna("")
y = df["quality_label"]


# ==========================================
# 3. Split Dataset
# ==========================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train_text))
print("Testing records:", len(X_test_text))
print()


# ==========================================
# 4. TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF vectorization completed!")
print("Features:", X_train.shape[1])
print()


# ==========================================
# 5. Train Logistic Regression Model
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("ML model trained successfully!")
print()


# ==========================================
# 6. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. Calculate Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")
print()


# ==========================================
# 8. Classification Report
# ==========================================

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. Confusion Matrix
# ==========================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()


# ==========================================
# 10. Save Model
# ==========================================

with open("job_description_model.pkl", "wb") as file:
    pickle.dump(model, file)


# ==========================================
# 11. Save TF-IDF Vectorizer
# ==========================================

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("=" * 60)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print("Created files:")
print("1. job_description_model.pkl")
print("2. tfidf_vectorizer.pkl")