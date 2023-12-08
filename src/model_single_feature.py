import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'transaction': ['Coffee at Cafe', 'Grocery shopping', 'Train ticket', 'Restaurant dining'],
    'category': ['Food & Beverage', 'Groceries', 'Transport', 'Food & Beverage']
}

df = pd.DataFrame(data)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['transaction'], df['category'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
