import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("master_dataset.csv")

# Clean 'Amount' column
df['Amount'] = (
    df['Amount']
    .astype(str)
    .str.replace(r'[^\d\.-]', '', regex=True)  # Remove all non-numeric characters (except dot and minus)
    .astype(float)
)

# Drop rows with missing labels
df = df.dropna(subset=['Category'])

# Clean text
df['Description'] = df['Description'].str.lower().str.strip()

# Encode target labels
le = LabelEncoder()
df['CategoryEncoded'] = le.fit_transform(df['Category'])

# Define features and target
X = df[['Description', 'Amount', 'Type']]
y = df['CategoryEncoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column transformer to handle different input types
preprocessor = ColumnTransformer(transformers=[
    ('desc', TfidfVectorizer(), 'Description'),
    ('type', OneHotEncoder(handle_unknown='ignore'), ['Type']),
    ('amount', 'passthrough', ['Amount'])
])

# Pipeline
model = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
model.fit(X_train, y_train)

# Evaluate
print("Model accuracy:", model.score(X_test, y_test))

joblib.dump(model, 'transaction_classifier.pkl')

