# GOOD-NLP: своя языковая модель как сервис

Учебный ML/Backend/MLOps проект: GPT-модель на русском языке, обученная с нуля, и сервис вокруг неё — от инференса в FastAPI до базы данных, авторизации, очереди задач и деплоя.

Модель маленькая (33.5M параметров): пишет грамматично и держит формат диалога, но по содержанию часто выдумывает. Цель проекта — не качество ответов, а полный цикл: собственная модель → нормальный сервис → продакшен.

## Статус

- [x] Токенизатор и предобучение модели (Kaggle)
- [x] SFT (instruction tuning)
- [x] Python-пакет с моделью и генерацией, консольный чат
- [x] FastAPI: `/health`, `/ready`, `POST /chat`
- [x] Docker-образ (CPU-версия PyTorch), docker compose
- [ ] PostgreSQL: история диалогов (SQLAlchemy, Alembic) — *в процессе*
- [ ] Пользователи и авторизация, лимиты запросов
- [ ] Минимальный фронтенд
- [ ] Нагрузочное тестирование, очередь задач, KV-кэш
- [ ] Деплой на VPS, CI/CD, HTTPS

## Модель

| | |
|---|---|
| Архитектура | GPT (decoder-only), Pre-LN, weight tying |
| Параметры | 33.5M: 8 слоёв, 8 голов, `n_embd=512` |
| Контекст | 256 токенов |
| Токенизатор | byte-level BPE, словарь 16 000, обучен на русской Википедии |
| Предобучение | ~2 ГБ русской Википедии, 40 000 шагов, val loss 2.95 |
| SFT | `IlyaGusev/ru_turbo_alpaca` (~29 000 примеров), loss только по ответу, val loss 2.33 |

Формат промпта: `Пользователь: {вопрос}\nАссистент:`. Ноутбуки с обучением — в `notebooks/`.

## Стек

Python 3.14 · PyTorch (CPU) · tokenizers · FastAPI · uvicorn · pydantic-settings · PostgreSQL · SQLAlchemy · Alembic · Docker · docker compose

## Структура

```
app/
├── main.py          # создание приложения, lifespan (загрузка модели)
├── config.py        # настройки из переменных окружения
├── api/             # эндпоинты: health.py, chat.py
├── schemas/         # pydantic-схемы запросов и ответов
├── ml/              # модель, загрузка, генерация (не зависит от FastAPI)
└── db/              # подключение к БД и модели таблиц
scripts/
└── chat_cli.py      # консольный чат для проверки модели и замера скорости
notebooks/           # обучение на Kaggle
models/              # файлы модели (не в git)
```

## Запуск

Файлы модели в репозиторий не входят (чекпоинт весит ~130 МБ). Положите `tokenizer.json` и `ckpt_sft.pt` в папку `models/`.

Переменные окружения: скопируйте `.env.example` в `.env` и задайте значения.

### Локально

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d db                       # PostgreSQL в контейнере
uvicorn app.main:app --reload --reload-dir app
```

Консольный чат без сервера:

```bash
python -m scripts.chat_cli
```

### В Docker

```bash
docker compose up --build
```

Документация API: http://localhost:8000/docs

## API

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | процесс жив |
| GET | `/ready` | модель загружена и БД доступна |
| POST | `/chat` | ответ модели на вопрос |

Пример запроса к `/chat`:

```json
{"question": "Как заварить чай?", "max_new_tokens": 100, "temperature": 0.7, "top_k": 50}
```

## Настройки

| Переменная | По умолчанию | Описание |
|---|---|---|
| `APP_CKPT_PATH` | `models/ckpt_sft.pt` | путь к чекпоинту |
| `APP_TOKENIZER_PATH` | `models/tokenizer.json` | путь к токенизатору |
| `APP_DATABASE_URL` | — (обязательна) | строка подключения к PostgreSQL |
| `APP_TORCH_THREADS` | авто | число потоков PyTorch |
| `APP_LOG_LEVEL` | `INFO` | уровень логов |

## Замеры

| Что | Значение |
|---|---|
| Скорость генерации (ноутбук, 6 потоков, без KV-кэша) | ~18 токенов/с |
| Размер Docker-образа | 327 МБ сжатый / 1.53 ГБ на диске |
| Память контейнера с моделью | ~325 МБ в простое, ~450 МБ при генерации |

Известные ограничения генерации, оставленные сознательно для последующей оптимизации: нет KV-кэша, выходной слой считается для всех позиций.

Ход работы, проблемы и решения — в [DEVLOG.md](DEVLOG.md).
