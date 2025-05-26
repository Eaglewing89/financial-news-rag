# Development Environment Guide

For instructions on installing and using `financial-news-rag` as a library, see `installation.md`.

This guide is for contributors and developers who want to work on the source code, run tests, or build new features.

## 1. Environment Setup

- **Python version:** Python 3.10 or higher
- **Clone the repository:**
  ```bash
  git clone https://github.com/Eaglewing89/financial-news-rag.git
  cd financial-news-rag
  ```
  (Or use SSH if preferred.)
- **Create virtual environment:**
  ```bash
  python -m venv venv
  ```
- **Activate virtual environment:**
  - On macOS and Linux:
    ```bash
    source venv/bin/activate
    ```
  - On Windows:
    ```bash
    .\venv\Scripts\activate
    ```
- **Install all dependencies (including dev tools):**
  ```bash
  pip install -r requirements.txt
  ```
  This will install all required packages for development, testing, and running the module, including tools for linting, formatting, and running example notebooks.
- **Install the project in editable mode:**
  ```bash
  pip install -e .
  ```
  This allows you to edit the source code and have changes reflected immediately without reinstalling.

## 2. Dependency Management

- All dependencies are listed in `requirements.txt`.
- To update dependencies, add or update package names in `requirements.txt` and re-run the install command above.
- For now, the project uses the latest compatible versions of each package. Version pinning (specifying exact versions) is not enforced, but can be added for reproducibility if needed.
- If the project grows, consider splitting requirements into `requirements.txt` (runtime) and `requirements-dev.txt` (development/testing tools).

## 3. Coding Standards and Patterns

- **Style:** Follow [PEP8](https://peps.python.org/pep-0008/) for code style.
- **Formatting:**
  - Format code with `black src/`
  - Sort imports with `isort src/`
  - Lint code with `flake8 src/`
- **Type hints:** Use Python type hints for all public functions and methods.
- **Docstrings:**
  - Write docstrings for all public modules, classes, and functions.
  - [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) or [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) are both acceptable.
  - Example (Google style):
    ```python
    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a (int): First integer.
            b (int): Second integer.

        Returns:
            int: The sum of a and b.
        """
    ```
- **Testing:**
  - Use `pytest` for all tests (see `tests/` directory).
  - Run tests with `pytest` and check coverage with `pytest --cov=financial_news_rag`.
- **Branching:**
  - Use descriptive branch names, e.g., `feature/feature-name`, `docs/foundation`.
- **Code Review:**
  - Self-review is encouraged. Collaborators may review their own code as needed.

## 4. Further Documentation

For more information, see the following documentation in the `docs/` folder:

- [Project Overview & Architecture](index.md)
- [Installation Guide](installation.md)
- [Usage Guide](usage_guide.md)
- [Configuration Reference](configuration.md)
- [API Reference](api_reference/)
- [Testing Guide](testing.md)
- [Development Guide](development.md) (this file)

---
For any questions or suggestions, please update this guide or contact the project maintainer.

