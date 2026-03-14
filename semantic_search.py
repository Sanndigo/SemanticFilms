"""
Семантический поиск фильмов с использованием sentence-transformers и FAISS.
Поддержка загрузки из CSV файлов (Кинопоиск, IMDb и др.).
"""

import json
import pickle
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticFilmSearch:
    """Класс для семантического поиска фильмов по описанию."""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        index_path: Optional[str] = None,
        data_path: Optional[str] = None,
        language: str = "ru"
    ):
        """
        Инициализация поискового движка.
        
        Args:
            model_name: Модель sentence-transformers (мультиязычная по умолчанию)
            index_path: Путь для сохранения/загрузки индекса FAISS
            data_path: Путь для сохранения/загрузки данных о фильмах
            language: Язык данных (ru/en)
        """
        self.language = language
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.movies: List[Dict[str, Any]] = []
        self.index_path = index_path or "film_index.faiss"
        self.data_path = data_path or "films_data.pkl"
    
    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Безопасное преобразование в int."""
        if value is None or value == "" or value == "-":
            return default
        try:
            return int(float(str(value).replace(",", ".").strip()))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Безопасное преобразование в float."""
        if value is None or value == "" or value == "-":
            return default
        try:
            return float(str(value).replace(",", ".").strip())
        except (ValueError, TypeError):
            return default
        
    def load_from_csv(
        self,
        csv_path: str,
        title_col: str = "title",
        plot_col: str = "description",
        genre_col: Optional[str] = "genre",
        rating_col: Optional[str] = "rating",
        year_col: Optional[str] = "year",
        duration_col: Optional[str] = None,
        poster_col: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8"
    ) -> None:
        """
        Загрузка фильмов из CSV файла.
        
        Args:
            csv_path: Путь к CSV файлу
            title_col: Название колонки с названием фильма
            plot_col: Название колонки с описанием
            genre_col: Название колонки с жанрами
            rating_col: Название колонки с рейтингом
            year_col: Название колонки с годом
            duration_col: Название колонки с длительностью (в минутах)
            poster_col: Название колонки с путем к постеру
            delimiter: Разделитель CSV
            encoding: Кодировка файла
        """
        print(f"Загрузка фильмов из {csv_path}...")
        
        self.movies = []
        
        with open(csv_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                plot = row.get(plot_col, "").strip()
                
                # Пропускаем фильмы без описания
                if not plot or len(plot) < 20:
                    continue
                
                movie = {
                    "title": row.get(title_col, "Unknown").strip(),
                    "overview": plot,
                    "genres": row.get(genre_col, "") if genre_col else "",
                    "rating": self._safe_float(row.get(rating_col, 0)) if rating_col else 0,
                    "year": self._safe_int(row.get(year_col, 0)) if year_col else 0,
                    "duration": self._safe_int(row.get(duration_col, 0)) if duration_col else 0,
                    "poster_path": row.get(poster_col, "") if poster_col else ""
                }
                
                self.movies.append(movie)
        
        print(f"Загружено {len(self.movies)} фильмов с описаниями")
    
    def load_from_json(
        self,
        json_path: str,
        title_key: str = "title",
        plot_key: str = "description",
        genre_key: Optional[str] = "genre",
        rating_key: Optional[str] = "rating",
        year_key: Optional[str] = "year",
        poster_key: Optional[str] = None
    ) -> None:
        """
        Загрузка фильмов из JSON файла.
        
        Args:
            json_path: Путь к JSON файлу
            title_key: Ключ с названием фильма
            plot_key: Ключ с описанием
            genre_key: Ключ с жанрами
            rating_key: Ключ с рейтингом
            year_key: Ключ с годом
            poster_key: Ключ с путем к постеру
        """
        print(f"Загрузка фильмов из {json_path}...")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Если данные в списке
        if isinstance(data, list):
            items = data
        # Если данные в ключе (например {"movies": [...]})
        elif isinstance(data, dict):
            # Пытаемся найти список в значениях
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    items = value
                    break
            else:
                items = []
        else:
            items = []
        
        self.movies = []
        for item in items:
            plot = item.get(plot_key, "").strip() if isinstance(item, dict) else ""
            
            if not plot or len(plot) < 20:
                continue
            
            movie = {
                "title": str(item.get(title_key, "Unknown")).strip(),
                "overview": plot,
                "genres": str(item.get(genre_key, "")) if genre_key else "",
                "rating": float(item.get(rating_key, 0) or 0) if rating_key else 0,
                "year": int(item.get(year_key, 0) or 0) if year_key else 0,
                "poster_path": str(item.get(poster_key, "")) if poster_key else ""
            }
            
            self.movies.append(movie)
        
        print(f"Загружено {len(self.movies)} фильмов с описаниями")
    
    def load_from_huggingface(
        self,
        dataset_name: str = "mt0rm0/movie_descriptors",
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> None:
        """
        Загрузка датасета из Hugging Face.
        
        Args:
            dataset_name: Название датасета
            split: Сплит датасета
            max_samples: Максимальное количество фильмов
        """
        from datasets import load_dataset
        
        print(f"Загрузка датасета {dataset_name}...")
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.movies = []
        for item in dataset:
            plot = item.get("overview", "").strip()
            
            if not plot or len(plot) < 20:
                continue
            
            movie = {
                "title": str(item.get("title", "Unknown")).strip(),
                "overview": plot,
                "genres": "",
                "rating": 0,
                "year": int(item.get("release_year", 0) or 0)
            }
            
            self.movies.append(movie)
        
        print(f"Загружено {len(self.movies)} фильмов с описаниями")
    
    def build_index(self, batch_size: int = 32) -> None:
        """Построение FAISS индекса для семантического поиска."""
        if not self.movies:
            raise ValueError("Сначала загрузите данные о фильмах")
        
        print("Создание эмбеддингов для фильмов...")
        
        # Тексты для эмбеддингов (название + описание)
        texts = [
            f"{movie['title']}. {movie['overview']}"
            for movie in self.movies
        ]
        
        # Генерируем эмбеддинги
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Создаем FAISS индекс
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Индекс создан: {self.index.ntotal} векторов")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Семантический поиск фильмов по запросу."""
        if self.index is None or not self.movies:
            raise ValueError("Индекс не создан. Загрузите данные и постройте индекс.")
        
        # Эмбеддинг запроса
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Поиск
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Результаты
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.movies):
                movie = self.movies[idx].copy()
                movie["relevance"] = float(score)
                results.append(movie)
        
        return results
    
    def save(self) -> None:
        """Сохранение индекса и данных."""
        if self.index is None:
            raise ValueError("Нечего сохранять.")
        
        faiss.write_index(self.index, self.index_path)
        
        with open(self.data_path, "wb") as f:
            pickle.dump(self.movies, f)
        
        print(f"Индекс сохранен: {self.index_path}")
        print(f"Данные сохранены: {self.data_path}")
    
    def load(self) -> bool:
        """Загрузка индекса и данных."""
        index_file = Path(self.index_path)
        data_file = Path(self.data_path)
        
        if not index_file.exists() or not data_file.exists():
            return False
        
        self.index = faiss.read_index(self.index_path)
        
        with open(self.data_path, "rb") as f:
            self.movies = pickle.load(f)
        
        print(f"Индекс загружен: {self.index.ntotal} векторов")
        print(f"Фильмов: {len(self.movies)}")
        return True


def print_results(results: List[Dict[str, Any]]) -> None:
    """Вывод результатов поиска."""
    if not results:
        print("Ничего не найдено")
        return
    
    print("\n" + "=" * 70)
    for i, movie in enumerate(results, 1):
        poster_url = ""
        if movie.get("poster_path"):
            if movie["poster_path"].startswith("http"):
                poster_url = movie["poster_path"]
            else:
                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
        
        # Форматируем длительность в часы:минуты
        duration = movie.get("duration", 0)
        duration_str = f"{duration // 60}ч {duration % 60}мин" if duration else "N/A"
        
        print(f"\n{i}. {movie['title']} ({movie.get('year', 'N/A')})")
        print(f"   Жанр: {movie.get('genres', 'N/A')}")
        print(f"   Рейтинг: {movie.get('rating', 'N/A')} | Длительность: {duration_str}")
        print(f"   Релевантность: {movie.get('relevance', 0):.4f}")
        print(f"   Описание: {movie.get('overview', '')[:250]}...")
        if poster_url:
            print(f"   Постер: {poster_url}")
    print("\n" + "=" * 70)
