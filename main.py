"""
Демонстрация семантического поиска фильмов.
Загрузка данных из CSV файла (Кинопоиск).

Использование:
    python main.py          # CLI версия
    python main.py web      # Веб-версия
"""

import sys
from pathlib import Path
from semantic_search import SemanticFilmSearch, print_results


def run_cli():
    """Запуск CLI версии."""
    # Инициализация
    searcher = SemanticFilmSearch(
        model_name="intfloat/multilingual-e5-base",
        index_path="film_index.faiss",
        data_path="films_data.pkl",
        language="ru"
    )

    # Пытаемся загрузить сохраненный индекс
    if searcher.load():
        print("✅ Индекс загружен из файла")
    else:
        print("📂 Создаем новый индекс...")

        # Используем файл kp_final.csv
        csv_path = "kp_final.csv"

        if Path(csv_path).exists():
            print(f"Загрузка из {csv_path}...")

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
        else:
            print(f"❌ Файл {csv_path} не найден!")
            return

        # Построение индекса
        searcher.build_index(batch_size=32)

        # Сохранение
        searcher.save()

    # Интерфейс консоли
    print("\n" + "=" * 70)
    print("Введите свой поисковый запрос (или 'exit' для выхода)")
    print("=" * 70)

    while True:
        try:
            query = input("\n🎬 Поиск: ").strip()

            if query.lower() in ("exit", "quit", "выход"):
                print("До свидания!")
                break

            if not query:
                continue

            results = searcher.search(query, top_k=5)
            print_results(results)

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def run_web():
    """Запуск веб-версии."""
    import uvicorn
    print("\n[WEB] Starting web interface...")
    print("Open in browser: http://127.0.0.1:8000")
    print("\nДля остановки нажмите Ctrl+C")
    print("-" * 70)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "web":
            run_web()
        else:
            run_cli()
    else:
        # По умолчанию запускаем CLI
        run_cli()


if __name__ == "__main__":
    main()

