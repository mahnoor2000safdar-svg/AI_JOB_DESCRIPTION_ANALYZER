import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "AI_Job_Description_Dataset_1000 (1).xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Dataset_1000",
    engine="openpyxl"
)


# ==========================================
# 2. Check Dataset
# ==========================================

print("Dataset columns:")
print(df.columns.tolist())
print()


# ==========================================
# 3. Check Missing Values
# ==========================================

print("Missing values:")
print(df.isnull().sum())
print()


# ==========================================
# 4. Text Cleaning Function
# ==========================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 5. Clean Job Descriptions
# ==========================================

df["clean_description"] = df["job_description"].apply(clean_text)


# ==========================================
# 6. Display Before and After Cleaning
# ==========================================

print("Example of cleaned text:")
print()

for i in range(5):
    print("ORIGINAL:")
    print(df["job_description"].iloc[i])

    print("\nCLEANED:")
    print(df["clean_description"].iloc[i])

    print("-" * 70)


# ==========================================
# 7. TF-IDF Vectorization
# ==========================================

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["clean_description"])

print("\nTF-IDF completed successfully!")
print("Number of records:", X.shape[0])
print("Number of features:", X.shape[1])


# ==========================================
# 8. Target Variable
# ==========================================

y = df["quality_label"]

print("\nTarget labels:")
print(y.value_counts())


# ==========================================
# 9. Save Processed Dataset
# ==========================================

df.to_csv(
    "processed_job_description_dataset.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")
print("File: processed_job_description_dataset.csv")