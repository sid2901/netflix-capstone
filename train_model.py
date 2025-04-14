import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("netflix_titles.csv")
df['rating'] = df['rating'].fillna("Unknown")
df['listed_in'] = df['listed_in'].fillna("")
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df = df.dropna(subset=['release_year'])
df['type_encoded'] = df['type'].map({'Movie': 1, 'TV Show': 0})
df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(", ")))

# Fit label encoder
label_enc = LabelEncoder()
df['rating_encoded'] = label_enc.fit_transform(df['rating'])

# Train model
X = df[['release_year', 'rating_encoded', 'genre_count']]
y = df['type_encoded']
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Save model and fitted encoder
joblib.dump(model, "best_type_model_v2.pkl")
joblib.dump(label_enc, "rating_encoder_v2.pkl")