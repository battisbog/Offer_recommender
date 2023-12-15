import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

# Sample dataset with additional features
data = {
    'transaction_name': ['Coffee at Cafe', 'Grocery shopping', 'Train ticket', 'Restaurant dining'],
    'amount': [5.0, 20.0, 15.0, 45.0],
    'location': ['New York', 'San Francisco', 'New York', 'Los Angeles'],
    'category': ['Food & Beverage', 'Groceries', 'Transport', 'Food & Beverage']
}

df = pd.DataFrame(data)

# Splitting the dataset
X = df[['transaction_name', 'amount', 'location']]
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Vectorize text, scale numerical features, and encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'transaction_name'),
        ('num', MinMaxScaler(), ['amount']),  # Replacing StandardScaler with MinMaxScaler
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
    ])

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test_transformed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
