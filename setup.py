from setuptools import setup, find_packages

setup(
    name="financial-news-rag",
    version="0.1.0",
    description="Retrieval Augmented Generation (RAG) system for financial news",
    author="Robert Örneving",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "chromadb",
        "google-generativeai",
        "requests",
        "python-dotenv",
        "nltk",
        "beautifulsoup4",
        "google-genai"
    ],
    python_requires=">=3.10",
)
