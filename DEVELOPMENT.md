# Development environment setup instructions

## 1. Activate virtual environment

```
source venv/bin/activate
```

## 2. Install dependencies

```
pip install -r requirements.txt
```

## 3. Lint and format code

- Format: `black src/`
- Sort imports: `isort src/`
- Lint: `flake8 src/`

## 4. Project structure (see project_spec.md for details)

```
financial-news-rag/
├── src/
│   └── financial_news_rag/
│       ├── __init__.py
│       ├── search.py
│       ├── embeddings.py
│       ├── data.py
│       ├── config.py
│       └── utils.py
├── tests/
│   ├── test_search.py
│   ├── test_embeddings.py
│   └── test_data.py
├── docs/
│   └── ...
├── examples/
│   ├── basic_search.ipynb
│   └── refresh_data.ipynb
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```
