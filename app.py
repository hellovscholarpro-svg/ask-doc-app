import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai

def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode("utf-8")]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        texts = text_splitter.create_documents(documents)

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        db = Chroma.from_documents(texts, embeddings)

        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4o-mini",
                temperature=0
            ),
            chain_type="stuff",
            retriever=retriever
        )

        return qa.run(query_text)

    return "Please upload a text file first."

st.set_page_config(page_title="Ask the Doc App")
st.title("Ask the Doc App")

uploaded_file = st.file_uploader("Upload an article", type="txt")

query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file
)

result = []

with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text)
    )

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )

    if submitted:
        if not openai_api_key.startswith("sk-") and not openai_api_key.startswith("sk-proj-"):
            st.error("Please enter a valid OpenAI API key.")
        else:
            try:
                with st.spinner("Calculating..."):
                    response = generate_response(uploaded_file, openai_api_key, query_text)
                    result.append(response)
                    del openai_api_key
            except openai.RateLimitError:
                st.error("OpenAI quota exceeded. Please check billing or use another API key.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if len(result):
    st.info(result[0])