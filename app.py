import streamlit as st
# To load api keys from .env file
from dotenv import load_dotenv
# To get pdf text
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# To convert & store chunks into vectorForm
# From OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# From HuggingFace
# from langchain.embeddings import HuggingFaceInstructEmbeddings

# Buffer Memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Language Model
from langchain.chat_models import ChatOpenAI

# HTML templete
from htmlTemplelates import css, bot_template, user_template

# To read pdf


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Define Text Spliter Properties
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Chunks in form of list
    chunks = text_splitter.split_text(text)
    return chunks


# convert list into embedding

# OpenAi
def get_vector_storeOpenAI(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore


# HuggingFace(Under Devlopment)
# '''
# def get_vector_storeHuggingFace(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")
#     vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorStore
# '''


# Conversation Chain
def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunk = get_text_chunks(raw_text)

                # create vector store
                # 1. Using OpenAi embedding model
                vectorStore = get_vector_storeOpenAI(text_chunk)

                # 2. HuggingFace Instructor
                # vectorStore = get_vector_storeHuggingFace(text_chunk)

                # Create conversation chain
                # session_state - if some state is change then st will reload the entire code, but we want to retrive conversation so we use this; This also make it similar to Global variable

                st.session_state.conversation = get_conversation_chain(vectorStore)
        


if __name__ == '__main__':
    main()
