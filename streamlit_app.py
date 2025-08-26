import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import requests
from scipy.sparse import issparse

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Header & Styling
# ----------------------------
st.markdown(
    """
    <style>
    body {background-color: #141414; color: #fff;}
    .stApp {background-color: #141414;}
    .movie-card {background-color: #1f1f1f; padding: 15px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=180)
st.markdown("<h1 style='color:#E50914;'>Movie Recommender</h1>", unsafe_allow_html=True)
st.caption("Built with TF-IDF + Cosine Similarity on IMDb Top-1000 dataset")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_recommender(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    return payload

# ----------------------------
# Fetch Poster & Plot from OMDb
# ----------------------------
@st.cache_data(show_spinner=False)
def get_movie_info(title: str):
    api_key = "7b3d7215"  # your OMDb API key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}&plot=short"
    try:
        res = requests.get(url).json()
        poster = res.get("Poster") if res.get("Poster") != "N/A" else None
        plot = res.get("Plot") if res.get("Plot") != "N/A" else "Plot not available."
        return poster, plot
    except Exception:
        return None, "Plot not available."

# ----------------------------
pkl_path = st.sidebar.text_input("Path to recommender pickle", "best_netflix_recommender.pkl")

payload = None
try:
    payload = load_recommender(pkl_path)
except Exception as e:
    st.error(f"Could not load pickle. Error: {e}")

if payload:
    vectorizer = payload['vectorizer']
    cosine_sim = payload['cosine_sim']
    indices = payload['indices']
    movies = payload['df']
    titles = movies['Series_Title'].astype(str).tolist()

    # ----------------------------
    # Sidebar Controls (cleaned)
    # ----------------------------
    st.sidebar.header("🔍 Find Movies")
    query = st.sidebar.selectbox("Select a movie", sorted(titles))
    topn = st.sidebar.slider("Number of recommendations", 5, 20, 10)
    min_rating = st.sidebar.slider("Minimum IMDb rating", 0.0, 10.0, 0.0, 0.1)

    # ----------------------------
    # Recommendation Logic (no genre filter)
    # ----------------------------
    def recommend(title, topn=10, min_rating=0.0):
        if title not in indices:
            return pd.DataFrame(columns=['title','genre','rating','similarity'])
        idx = indices[title]
        sims = cosine_sim[idx]
        sims = sims.toarray().ravel() if issparse(sims) else np.asarray(sims).ravel()
        sim_scores = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: topn+50]
        movie_indices = [i for i,_ in sim_scores]
        result = movies.iloc[movie_indices].copy()
        result['similarity'] = [s for _, s in sim_scores]
        if min_rating > 0:
            if 'rating' in result.columns:
                result = result[result['rating'].fillna(0) >= min_rating]
        return result.head(topn)[['Series_Title','Genre','IMDB_Rating','Director','similarity']]

    # ----------------------------
    # Progress Loader
    # ----------------------------
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.subheader("🎥 Generating Recommendations...")

    for i in range(80):
        time.sleep(0.01)
        progress_bar.progress(i+1)

    recs = recommend(query, topn=topn, min_rating=min_rating)

    for i in range(80, 100):
        time.sleep(0.01)
        progress_bar.progress(i+1)

    progress_bar.empty()
    status_text.empty()

    # ----------------------------
    # Display Results
    # ----------------------------
    if recs.empty:
        st.warning("No recommendations found. Try adjusting filters.")
    else:
        st.success(f"Top {len(recs)} movies similar to **{query}**:")
        
        for _, row in recs.iterrows():
            with st.container():
                col1, col2 = st.columns([1,3])
                with col1:
                    poster_url, plot = get_movie_info(row['Series_Title'])
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image(f"https://via.placeholder.com/150x220.png?text={row['Series_Title']}", use_container_width=True)
                with col2:
                    st.markdown(f"<h3 style='color:#E50914;'>{row['Series_Title']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"**Genre:** {row['Genre']}")
                    st.markdown(f"**Director:** {row['Director']}")
                    st.markdown(f"⭐ IMDb Rating: {row['IMDB_Rating']}/10")
                    st.markdown(f"**Plot:** {plot}")
                    st.progress(min(float(row['similarity']), 1.0))
            st.divider()

    st.caption("Made with ❤️ using Streamlit, scikit-learn & IMDb dataset.")
else:
    st.info("Please provide a valid recommender pickle in the sidebar.")
