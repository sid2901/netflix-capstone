import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load model and encoder
model = joblib.load("best_type_model_v2.pkl")
encoder = joblib.load("rating_encoder_v2.pkl")

# Load dataset
df = pd.read_csv("netflix_titles.csv")
df['description'] = df['description'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')
df['rating'] = df['rating'].fillna('Unknown')
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df = df.dropna(subset=['release_year'])

# Keyword map
keyword_map = {
    "romcom": ["Romantic Movies", "Comedies"],
    "action": ["Action & Adventure", "TV Action & Adventure"],
    "drama": ["Dramas", "TV Dramas"],
    "crime": ["Crime TV Shows", "Crime Movies"],
    "thriller": ["Thrillers", "TV Thrillers"],
    "fantasy": ["Sci-Fi & Fantasy", "TV Sci-Fi & Fantasy"],
    "romance": ["Romantic Movies", "Romantic TV Shows"],
    "comedy": ["Comedies", "TV Comedies"],
    "documentary": ["Documentaries", "Docuseries"]
}

def recommend_titles(keyword, content_type, min_year, max_year, top_n=5):
    genres = keyword_map.get(keyword.lower(), [])
    filtered = df[df['listed_in'].apply(lambda x: any(g in x for g in genres))]
    if content_type in ['Movie', 'TV Show']:
        filtered = filtered[filtered['type'] == content_type]
    filtered = filtered[(filtered['release_year'] >= min_year) & (filtered['release_year'] <= max_year)]

    if filtered.empty:
        return pd.DataFrame()

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(filtered["description"])
    user_input = tfidf.transform([keyword])
    cosine_scores = linear_kernel(user_input, tfidf_matrix).flatten()

    top_indices = cosine_scores.argsort()[-top_n:][::-1]
    return filtered.iloc[top_indices][['title', 'listed_in', 'description', 'rating', 'type']]

# Streamlit UI
st.set_page_config(page_title="Netflix Capstone App", layout="centered")
st.title("🎓 Netflix Capstone App")

mode = st.radio("Choose Mode:", ["🎬 Predict Content Type", "🎯 Recommend Titles"])

if mode == "🎬 Predict Content Type":
    st.subheader("Input Features")
    year = st.slider("Release Year", 1950, 2025, 2020)
    rating = st.selectbox("Rating", df['rating'].dropna().unique().tolist())
    rating_enc = encoder.transform([rating])[0]
    genre_count = st.slider("Number of Genres", 1, 5, 2)

    if st.button("Predict Type"):
        pred = model.predict([[year, rating_enc, genre_count]])[0]
        st.success("Prediction: 🎥 Movie" if pred else "Prediction: 📺 TV Show")

elif mode == "🎯 Recommend Titles":
    st.subheader("Recommendation Filters")
    keyword = st.text_input("Enter a Genre Keyword (e.g., romcom, crime, action)")
    content_type = st.radio("Content Type", ["Any", "Movie", "TV Show"])
    year_range = st.slider("Release Year Range", 1950, 2025, (2015, 2023))

    if st.button("Recommend"):
        recs = recommend_titles(keyword, content_type if content_type != "Any" else None,
                                year_range[0], year_range[1], top_n=5)
        if recs.empty:
            st.warning("No matching content found.")
        else:
            for _, row in recs.iterrows():
                msg = "<b>{}</b><br>{}<br>{}<br>{} | {}".format(
                    row['title'], row['listed_in'], row['description'], row['rating'], row['type']
                )
                st.markdown(msg, unsafe_allow_html=True)