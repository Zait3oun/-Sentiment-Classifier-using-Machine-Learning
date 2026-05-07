import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# قراءة الداتا
df = pd.read_csv("dataset.csv")

# الأعمدة
X = df["text"]
y = df["sentiment"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

# الموديل
model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

# حفظ الملفات
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Saved Successfully")