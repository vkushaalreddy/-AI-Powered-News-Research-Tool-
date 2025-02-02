# AI-Powered News Research Tool 

 AI-Powered News Research Tool  is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.

![image](https://github.com/user-attachments/assets/bd5d4ad8-0fc8-4b53-b967-1d5507f35b55)


## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Extracts and processes news articles from given URLs using web scraping.
- Process article content through LangChain's UnstructuredURL Loader
- Stores and retrieves article embeddings using a vector database (e.g., FAISS, Pinecone, ChromaDB) for efficient querying.
- Uses OpenAI's LLM for answering questions based on news content.
- Supports semantic search to find related articles based on query context.

## TechStack

- Python – Core language
- OpenAI GPT API – LLM for news analysis
- UnstructuredUrLLoader – Web scraping for fetching news content
- Vector Database (FAISS / Pinecone / ChromaDB) – Efficient semantic search
- Streamlit / Flask – For UI (if applicable)



## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles
2 years ago




## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- index.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.
