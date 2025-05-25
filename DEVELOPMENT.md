# Development Environment Guide

## 1. Environment Setup

- **Python version:** 3.12.10 (recommended)
- **Create virtual environment:**
  ```bash
  python -m venv venv
  ```
- **Activate virtual environment:**
  ```bash
  source venv/bin/activate
  ```
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
  This will install all required packages for both development and running the module. (Currently, there is a single `requirements.txt` for all dependencies.)

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

## 4. Project Structure

See `docs/project_spec.md` for detailed structure and module responsibilities.

## 5. Additional Resources

- [Project Specification](docs/project_spec.md)
- [Technical Design](docs/technical_design.md)
- [Testing Guide](docs/testing.md)

---
For any questions or suggestions, please update this guide or contact the project maintainer.
