# Installation

This guide provides instructions on how to set up and install the `financial-news-rag` library for your project.

## Prerequisites

- Python 3.10 or higher.
- `pip` for installing Python packages.
- Git (optional, primarily for developers or if installing directly from the repository).


## Installing `financial-news-rag`

You can easily install the `financial-news-rag` package in your own Python project without needing to clone the repository.

### Method 1: Install directly from GitHub

Run this command in your terminal:
```bash
pip install git+https://github.com/Eaglewing89/financial-news-rag.git
```
This will install the latest version of the package and all required dependencies.

### Method 2: Install from a downloaded ZIP file

1. Go to the [project's GitHub page](https://github.com/Eaglewing89/financial-news-rag).
2. Click the green "Code" button and select "Download ZIP".
3. Save the ZIP file to your computer (e.g., `financial-news-rag-main.zip`).
4. Install the package using pip:
    ```bash
    pip install /path/to/your/financial-news-rag-main.zip
    ```
    (Replace `/path/to/your/` with the actual path to the ZIP file.)



## Common Next Steps (for all installation methods)

### API Key Configuration

The `financial-news-rag` system requires API keys for two services: EODHD (for financial data and news) and Google Gemini (for AI-powered re-ranking and embedding).

You need to create a `.env` file in the root of your project directory.

#### a. EODHD API Key

1.  Sign up for an account on the [EODHD website](https://eodhd.com/).
2.  Retrieve your API token from your account dashboard.
3.  Add the following line to your `.env` file, replacing `your_eodhd_api_key_here` with your actual key:
    ```env
    EODHD_API_KEY=your_eodhd_api_key_here
    ```

#### b. Google Gemini API Key

1.  Go to [Google AI Studio (formerly MakerSuite)](https://ai.google.dev/) or the Google Cloud Console to obtain your API key for the Gemini models.
2.  Ensure the Gemini API is enabled for your project if using Google Cloud.
3.  Add the following line to your `.env` file, replacing `your_gemini_api_key_here` with your actual key:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

**Example `.env` file:**
```env
# .env
EODHD_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=yyyyyyyyyyyyyyyyyyyyyyyyy
```

**Note:** Both services offer a free tier.

### Verify Installation and Explore Usage

-   The primary way to verify is to import the package in your Python code (`from financial_news_rag import FinancialNewsRAG`) and use its functionalities as described in `usage_guide.md`.
-   To explore detailed examples provided with the package:
    1.  You can clone the [GitHub repository](https://github.com/Eaglewing89/financial-news-rag.git) separately to access the `examples/` directory and the `financial_news_rag_example.ipynb` notebook.
    2.  Alternatively, refer to the `usage_guide.md` for comprehensive API usage details and code snippets.

---

You should now have a working installation of the `financial-news-rag` project. For detailed usage, please refer to the `usage_guide.md`.
