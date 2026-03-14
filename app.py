"""
Веб-приложение для семантического поиска фильмов.
FastAPI + Tailwind CSS
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Any, Dict

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Инициализация приложения
app = FastAPI(title="Semantic Film Search")

# Глобальные переменные для поискового движка
searcher = None
model = None
index = None
movies = []


class SearchQuery(BaseModel):
    query: str
    top_k: int = 20


class Film(BaseModel):
    id: int
    title: str
    year: int
    rating: float
    duration: int
    genres: str
    overview: str
    poster_path: str
    relevance: Optional[float] = None


def init_searcher():
    """Инициализация поискового движка."""
    global searcher, model, index, movies
    
    index_path = "film_index.faiss"
    data_path = "films_data.pkl"
    
    if not Path(index_path).exists() or not Path(data_path).exists():
        print("[ERROR] Index not found. Please run main.py first to create the index.")
        return False
    
    # Загружаем модель
    print("Loading model...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    
    # Загружаем индекс
    print("Loading FAISS index...")
    index = faiss.read_index(index_path)
    
    # Загружаем данные
    print("Loading film data...")
    with open(data_path, "rb") as f:
        movies = pickle.load(f)
    
    print(f"[SUCCESS] Loaded {len(movies)} films")
    return True


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске."""
    init_searcher()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница."""
    return FileResponse("templates/index.html")


@app.post("/api/search")
async def search_films(query: SearchQuery):
    """Поиск фильмов по запросу."""
    if model is None or index is None or not movies:
        raise HTTPException(status_code=500, detail="Поисковый движок не инициализирован")
    
    # Создаем эмбеддинг запроса
    query_embedding = model.encode(
        [query.query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Поиск
    scores, indices = index.search(query_embedding, query.top_k)
    
    # Формируем результаты
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(movies):
            movie = movies[idx]
            results.append({
                "id": int(idx),
                "title": movie.get("title", "Unknown"),
                "year": int(movie.get("year", 0)),
                "rating": float(movie.get("rating", 0)),
                "duration": int(movie.get("duration", 0)),
                "genres": movie.get("genres", ""),
                "overview": movie.get("overview", ""),
                "poster_path": movie.get("poster_path", ""),
                "relevance": float(score)
            })
    
    return {"results": results}


@app.get("/api/film/{film_id}")
async def get_film(film_id: int):
    """Получение информации о фильме."""
    if film_id < 0 or film_id >= len(movies):
        raise HTTPException(status_code=404, detail="Фильм не найден")
    
    movie = movies[film_id]
    return {
        "id": film_id,
        "title": movie.get("title", "Unknown"),
        "year": movie.get("year", 0),
        "rating": movie.get("rating", 0),
        "duration": movie.get("duration", 0),
        "genres": movie.get("genres", ""),
        "overview": movie.get("overview", ""),
        "poster_path": movie.get("poster_path", ""),
        "countries": movie.get("countries", ""),
    }


@app.get("/api/popular")
async def get_popular(limit: int = 20):
    """Популярные фильмы (по рейтингу)."""
    if not movies:
        raise HTTPException(status_code=500, detail="Данные не загружены")
    
    # Сортируем по рейтингу
    sorted_movies = sorted(
        enumerate(movies),
        key=lambda x: x[1].get("rating", 0),
        reverse=True
    )[:limit]
    
    results = []
    for idx, movie in sorted_movies:
        results.append({
            "id": idx,
            "title": movie.get("title", "Unknown"),
            "year": movie.get("year", 0),
            "rating": movie.get("rating", 0),
            "duration": movie.get("duration", 0),
            "genres": movie.get("genres", ""),
            "overview": movie.get("overview", ""),
            "poster_path": movie.get("poster_path", ""),
        })
    
    return {"results": results}


@app.get("/api/genres")
async def get_genres():
    """Получение списка жанров."""
    if not movies:
        raise HTTPException(status_code=500, detail="Данные не загружены")
    
    genres_set = set()
    for movie in movies:
        genres = movie.get("genres", "")
        if genres:
            for genre in genres.split(","):
                genres_set.add(genre.strip())
    
    return {"genres": sorted(list(genres_set))}


@app.get("/api/genre/{genre_name}")
async def get_by_genre(genre_name: str, limit: int = 20):
    """Фильмы по жанру."""
    if not movies:
        raise HTTPException(status_code=500, detail="Данные не загружены")
    
    # Фильтруем по жанру
    filtered = []
    for idx, movie in enumerate(movies):
        genres = movie.get("genres", "").lower()
        if genre_name.lower() in genres:
            filtered.append({
                "id": idx,
                "title": movie.get("title", "Unknown"),
                "year": movie.get("year", 0),
                "rating": movie.get("rating", 0),
                "duration": movie.get("duration", 0),
                "genres": movie.get("genres", ""),
                "overview": movie.get("overview", ""),
                "poster_path": movie.get("poster_path", ""),
            })
    
    # Сортируем по рейтингу
    filtered.sort(key=lambda x: x["rating"], reverse=True)
    
    return {"results": filtered[:limit]}
