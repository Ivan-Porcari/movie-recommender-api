from fastapi import FastAPI

app = FastAPI(
    title="Movie Recommender API",
    description="API de recomendación de películas con filtrado colaborativo",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Movie Recommender API funcionando 🎬"}

@app.get("/health")
def health_check():
    return {"status": "ok"}