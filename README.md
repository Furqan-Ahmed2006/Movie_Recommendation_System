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

