"""
Скрипт для отдельного запуска процесса индексации фильмов.
"""

import sys
from pathlib import Path
from semantic_search import SemanticFilmSearch


def build_film_index():
    """Создание индекса фильмов."""
    print("🎬 Начинаем процесс индексации фильмов...")
    
    # Инициализация поискового движка
    searcher = SemanticFilmSearch(
        model_name="intfloat/multilingual-e5-base",
        index_path="film_index.faiss",
        data_path="films_data.pkl",
        language="ru"
    )
    
    # Проверяем наличие CSV файла
    csv_path = "kp_final.csv"
    if not Path(csv_path).exists():
        print(f"❌ Файл {csv_path} не найден!")
        return False
    
    print(f"📁 Загрузка фильмов из {csv_path}...")
    
    # Загрузка из CSV с Кинопоиска
    searcher.load_from_csv(
        csv_path=csv_path,
        title_col="name_rus",
        plot_col="description",
        genre_col="genres",
        rating_col="kp_rating",
        year_col="movie_year",
        duration_col="movie_duration",
        poster_col="poster",
        delimiter=",",
        encoding="utf-8"
    )
    
    if not searcher.movies:
        print("❌ Не удалось загрузить фильмы из CSV файла")
        return False
    
    print(f"📊 Создание индекса для {len(searcher.movies)} фильмов...")
    
    # Построение индекса
    searcher.build_index(batch_size=32)
    
    # Сохранение индекса
    print("💾 Сохранение индекса и данных...")
    searcher.save()
    
    print("✅ Индексация завершена успешно!")
    print(f"📈 Всего проиндексировано: {searcher.index.ntotal} фильмов")
    
    return True


def main():
    """Основная функция запуска индексации."""
    try:
        success = build_film_index()
        if success:
            print("\n🎉 Процесс индексации успешно завершен!")
        else:
            print("\n❌ Процесс индексации завершился с ошибкой!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Процесс индексации был прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка при индексации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()