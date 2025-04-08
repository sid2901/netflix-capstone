
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the Netflix dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df['content_features'] = df['listed_in'] + ' ' + df['description']
    return df

df = load_data()

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content_features'])

# Keyword expansion map
keyword_expansion = {
    "romcom": "romantic comedy love story light-hearted humorous relationship dating feel-good",
    "action": "action-packed thrilling intense fight chase superhero war crime high-stakes adrenaline",
    "sci-fi": "science fiction futuristic space aliens time-travel advanced technology dystopian",
    "drama": "emotional serious story character-driven life realistic deep thoughtful intense",
    "comedy": "funny humorous stand-up light-hearted sitcom jokes witty slapstick quirky",
    "horror": "scary thriller suspenseful haunted ghosts supernatural jump scares creepy dark",
    "crime": "criminal investigation detective police mystery heist courtroom underworld gang",
    "fantasy": "magic mythical creatures kingdoms wizards dragons enchanted world spells",
    "romance": "love romantic emotional couple relationship heartwarming dating affection",
    "thriller": "suspense mystery psychological intense crime fast-paced unexpected twist",
    "documentary": "real-life true story history biography investigative informative fact-based",
    "animation": "animated family-friendly cartoons kids colorful creative fun voice-acted",
    "family": "family-friendly heartwarming parenting kids bonds life lessons values",
    "mystery": "mysterious unknown clues solve puzzle twist detective secret hidden",
    "adventure": "epic journey quest explore travel thrilling exciting expedition",
    "biography": "true story real life famous people inspirational journey career",
    "historical": "past historical events period costume war politics empire revolution",
    "teen": "teenage high school coming-of-age romance friends social awkward youth",
    "musical": "music dance performance singing songs stage artistic emotional",
    "western": "cowboys guns wild west frontier standoff ranch outlaw saloon",
    "war": "military combat battles strategy historical soldiers army violence survival",
    "sports": "athletes competition games victory defeat team motivation training"
}

# Recommendation function
def recommend_from_keyword(user_keyword, num_recommendations=10):
    expanded_input = keyword_expansion.get(user_keyword.lower(), user_keyword)
    user_vector = tfidf.transform([expanded_input])
    sim_scores = linear_kernel(user_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]
    return df.iloc[top_indices][['title', 'listed_in', 'description']].reset_index(drop=True)

# Streamlit UI
st.title("🎬 Netflix Content Recommender")

user_input = st.text_input("Enter a genre or keyword (e.g., romcom, sci-fi, drama):", "romcom")

if st.button("Get Recommendations"):
    with st.spinner("Finding content you'll love..."):
        results = recommend_from_keyword(user_input)
        if not results.empty:
            for _, row in results.iterrows():
                st.subheader(row['title'])
                st.write("**Genres:**", row['listed_in'])
                st.write(row['description'])
                st.markdown("---")
        else:
            st.warning("No recommendations found.")
