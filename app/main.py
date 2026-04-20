from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.recommender import Recommender

recommender = Recommender()
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esto se ejecuta UNA SOLA VEZ al arrancar
    print("🎬 Entrenando modelos...")
    recommender.train()
    print("✅ Modelos listos!")
    yield
    # Esto se ejecuta al cerrar la API
    print("👋 Cerrando API...")

app = FastAPI(
    title="Movie Recommender API",
    description="API de recomendación de películas con filtrado colaborativo",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"message": "Movie Recommender API funcionando 🎬"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/recommendations/content/{title}")
def content_recommendations(title: str, n: int = 10):
    results = recommender.get_content_recommendations(title, n)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    return results.to_dict('records')

@app.get("/recommendations/collaborative/{user_id}")
def collaborative_recommendations(user_id: int, n: int = 10):
    results = recommender.get_collaborative_recommendations(user_id, n)
    if results is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return results.to_dict('records')