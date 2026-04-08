# 🎬 Hybrid Movie Recommendation System (ML + TMDB API)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Overview
This is a professional, highly scalable **Hybrid Movie Recommendation System**. It connects a custom-trained **Natural Language Processing (NLP)** machine learning model with the live **TMDB (The Movie Database) API** to deliver seamless recommendations.

Unlike hard-coded basic machine learning structures that crash when asked about a brand-new movie, this system was architected with an **Intelligent Fallback Engine**:
- **🧠 Primary Engine:** If a movie exists within the trained dataset, the system uses custom TF-IDF Vectorization and Cosine Similarity to find hyper-accurate relations based on movie metadata. 
- **🛟 Fallback Engine:** If a user searches for a movie not found in the trained dataset (e.g., a movie released just yesterday), the system dynamically routes the request to TMDB's Recommendation Engine without ever breaking the user experience.

## ✨ Features
- **Scalable Microservice Architecture:** Decoupled Backend (FastAPI) and Frontend (Streamlit).
- **Infinite Scalability:** Users can request exactly how many recommendations they want (up to 20 per request).
- **Live Poster Fetching:** Pulls high-quality movie posters globally.
- **Fail-Safe UX:** Ensures users *never* hit a "Movie Not Found" error for valid titles.

## 🚀 How to Run Locally

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your TMDB API Key:**
   Create a `.env` file in the root folder and add:
   ```env
   api=YOUR_TMDB_API_KEY
   ```

3. **Start the FastAPI Backend (Service 1):**
   ```bash
   uvicorn api:app --reload
   ```

4. **Start the Streamlit Frontend UI (Service 2):**
   Open a secondary terminal:
   ```bash
   streamlit run app.py
   ```

## ☁️ How to Deploy on Render

This project requires **Two separate Web Services** on Render because it models a true microservice architecture. Do not deploy them as a "Background Worker".

### 1. Deploy the Backend (FastAPI)
1. Go to [Render.com](https://render.com/) -> **New** -> **Web Service**.
2. Connect your GitHub repository.
3. Name it something like `movie-backend-api`.
4. **Environment:** `Python 3`
5. **Build Command:** `pip install -r requirements.txt`
6. **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
7. _Wait for it to deploy and copy the live URL (e.g. `https://movie-backend-api.onrender.com`)._

### 2. Connect the UI
Before deploying the Streamlit UI, edit line 5 of `app.py`:
Change `API_URL = "http://127.0.0.1:8000/recommend/"` to your new live backend URL:
`API_URL = "https://movie-backend-api.onrender.com/recommend/"`

### 3. Deploy the Frontend (Streamlit)
1. Go to **Render.com** -> **New** -> **Web Service**.
2. Connect the exact same GitHub repository.
3. Name it something like `movie-frontend-ui`.
4. **Environment:** `Python 3`
5. **Build Command:** `pip install -r requirements.txt`
6. **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
7. **Deploy!**

Your full-stack ML application will now be live on the internet! 
