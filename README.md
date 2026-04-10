# 🎬 Movie Recommender API

A REST API that recommends movies based on content filtering and collaborative filtering, built with FastAPI and scikit-learn.

## 🚀 Tech Stack

- **FastAPI** - API framework
- **scikit-learn** - Recommendation models
- **pandas** - Data processing
- **MovieLens** - Dataset (small version)

## 📦 Setup

```bash
# Clone the repository
git clone https://github.com/Ivan-Porcari/movie-recommender-api.git
cd movie-recommender-api

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

## 📍 Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/recommendations/{movie_id}` | Get movie recommendations *(coming soon)* |

## 🗺️ Roadmap

- [x] Project setup & FastAPI structure
- [x] Data exploration (MovieLens dataset)
- [ ] Content-based filtering model
- [ ] Collaborative filtering model
- [ ] Recommendation endpoints