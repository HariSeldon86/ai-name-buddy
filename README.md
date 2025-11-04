# AI Name Buddy

AI Name Buddy is a command-line tool that leverages the power of Large Language Models (LLMs) and a sophisticated agentic system to help you create standardized technical abbreviations and descriptions for new keywords. This tool is designed to ensure consistency and relevance in your project's naming conventions.

## Key Features

*   **AI-Powered Name Suggestion**: Utilizes a local LLM to generate contextually relevant abbreviations and descriptions.
*   **Agentic System**: Employs a LangChain agent that intelligently interacts with a set of tools to validate and generate unique names.
*   **Semantic Search**: Leverages a ChromaDB vector store for semantic search, enabling the agent to find similar keywords and generate more accurate suggestions.
*   **Local LLM Support**: Natively supports local LLMs through Ollama, giving you full control over your models and data.

## How It Works

The AI Name Buddy employs a sophisticated agentic workflow to generate high-quality name suggestions:

1.  **Data Initialization**: On the first run, the application initializes its knowledge base by:
    *   Populating an SQLite database from a `Dictionary.json` file.
    *   Creating a ChromaDB vector store from the records in the SQLite database.
2.  **User Input**: The user provides a new keyword for which they need an abbreviation and description.
3.  **Keyword Validation**: The agent first checks if the keyword already exists in the project's dictionary to avoid duplicates.
4.  **Name Suggestion**: If the keyword is new, the agent uses a `name_suggestion_tool`. This tool queries the ChromaDB vector store to find existing keywords that are semantically similar to the user's input. The retrieved examples are then used to generate a candidate abbreviation and description that align with the established naming conventions.
5.  **Abbreviation Uniqueness Check**: The agent then verifies if the suggested abbreviation is already in use.
6.  **Iterative Refinement**: If the abbreviation is not unique, the agent calls the `name_suggestion_tool` again, this time providing additional context to guide the LLM in generating a different abbreviation.
7.  **Final Suggestion**: Once a unique and contextually relevant abbreviation is generated, the agent presents the final suggestion to the user, who can then choose to save it to the project's dictionary.

## Getting Started

### Prerequisites

*   Python 3.12+
*   A running instance of Ollama with the required LLM and embedding models.
*   `uv` package and project manager is recommended.

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/HariSeldon86/ai-name-buddy.git
    cd ai-name-buddy
    ```

2.  Install the dependencies:

    ```bash
    # Using uv (recommended)
    uv sync --locked

    # Using pip
    # pip install -r requirements.txt
    ```

### Running the Application

To start the AI Name Buddy, run the `main.py` script:

```bash
# Using uv (recommended)
uv run main.py

# Using python
#python main.py
```

The application will prompt you to enter a new keyword.

## Configuration

You can easily configure the LLM and embedding models used by the application by modifying the `config.py` file. By default, the application is configured to use the following models:

*   **LLM Model**: `gemma3n:e4b`
*   **Embedding Model**: `embeddinggemma:300m`

Feel free to replace these with any other models available in your Ollama instance.

> **NOTE:** Changing **Embedding Model** requires regenerating the `vectorstore`.
