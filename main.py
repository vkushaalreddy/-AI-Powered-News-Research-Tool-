import os
import streamlit as st
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

# FAISS index directory
faiss_index_path = "faiss_index"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=[url for url in urls if url.strip()])
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Embeddings and FAISS Index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index along with metadata
        vectorstore_openai.save_local(faiss_index_path)
        st.success("Processing completed! You can now ask questions.")

    except Exception as e:
        st.error(f"An error occurred while processing URLs: {e}")

# Query Section
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(faiss_index_path):
        try:
            # Load FAISS index along with metadata
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_index_path, embeddings)

            # Retrieval QA Chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display Answer
            st.header("Answer")
            st.write(result.get("answer", "No answer found."))

            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
    else:
        st.warning("No processed data found. Please enter URLs and process first.")
