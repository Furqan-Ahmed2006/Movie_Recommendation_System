import os
import pickle
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import linear_kernel
import requests
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Hybrid Movie Recommendation System")

TMDB_API_KEY = os.getenv("api")

print("Loading NLP & ML model files...")
try:
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    print("ML models loaded successfully! 🚀")
except Exception as e:
    print(f"Error loading ML files: {e}")


def search_movie_on_tmdb(movie_title):
    """Searches TMDB to get the exact standard movie title and ID."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data.get("results"):
        return data["results"][0]["id"], data["results"][0]["title"]
    raise ValueError(f"Could not find '{movie_title}' on TMDB.")

def get_poster(movie_title):
    """Fetches just the poster for a specific movie."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    try:
        response = requests.get(url).json()
        if response.get("results") and response["results"][0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + response["results"][0]["poster_path"]
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

def get_ml_recommendations(movie_title, count):
    """ Custom Machine Learning recommendation logic (NLP, TF-IDF, Cosine Similarity)"""
    idx = indices[movie_title]
    
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:count+1]
    movie_indices = [i[0] for i in sim_scores]
    
    col_name = 'title' if 'title' in df.columns else df.columns[0]
    recommended_titles = df[col_name].iloc[movie_indices].tolist()
    
    results = []
    for title in recommended_titles:
        results.append({
            "title": str(title),
            "poster_url": get_poster(str(title))
        })
    return results

def get_tmdb_fallback_recommendations(movie_id, count):
    """TMDB API Fallback for new movies not in your dataset"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}"
    response = requests.get(url).json()
    
    results = []
    if response.get("results"):
        for movie in response["results"][:count]:
            poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Poster"
            results.append({
                "title": movie.get("title", "Unknown"),
                "poster_url": poster
            })
    return results


@app.get("/")
def home():
    return {"message": "Hybrid ML & TMDB Recommendation API is running!"}

@app.get("/recommend/")
def recommend(movie_title: str, count: int = 5):
    try:
        movie_id, exact_title = search_movie_on_tmdb(movie_title)
        
        
        if exact_title in indices:
            recs = get_ml_recommendations(exact_title, count)
            engine_used = "Custom Machine Learning Model (TF-IDF & Cosine Similarity)"
        elif movie_title in indices:
            recs = get_ml_recommendations(movie_title, count)
            engine_used = "Custom Machine Learning Model (TF-IDF & Cosine Similarity)"
        else:
            recs = get_tmdb_fallback_recommendations(movie_id, count)
            engine_used = "TMDB Fallback API Engine"
            
        return {
            "movie": exact_title,
            "engine_used": engine_used, 
            "recommendations": recs
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
