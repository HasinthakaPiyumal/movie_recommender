from fastapi import FastAPI, Query, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from movie_recommender import MovieRecommender
import os
from starlette.status import HTTP_403_FORBIDDEN

app = FastAPI()
print("Starting Movie Recommender API...")
mr = MovieRecommender()
print("Movie Recommender initialized.")

# API Key setup
API_KEY = os.getenv("SYSTEM_KEY", "ns53gnoeu8d9opRYEoiPpeGfDMTGtnlvhHyEQYk56e4R7fsJe7f2")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, 
        detail="Could not validate API key"
    )

@app.get("/get-movies")
def predict(
    favorite: str = Query(..., description="Name of the movie to predict"),
    n_recommendations: int = Query(..., description="Requested number of movie recommendations", ge=1, le=100),
    api_key: APIKey = Depends(get_api_key)
):
    recommended_movies = mr.get_recommended_movies(favorite, n_recommendations)
    if isinstance(recommended_movies, str):
        return {"error": recommended_movies}
    return {"prediction": favorite, "recommended_movies": recommended_movies}