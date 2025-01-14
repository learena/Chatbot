####################################################################
#                         import
####################################################################

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.schema import format_document
from dotenv import load_dotenv
load_dotenv()

token1=os.getenv("OPENAI_KEY")
token2=os.getenv("GOOGLE_KEY")
token3=os.getenv("COHERE_KEY")

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# GitPython for pushing vectorstores to GitHub
from git import Repo

# Streamlit for app interface
import streamlit as st

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
]

dict_welcome_message = {
    "Italiano": "Come posso assistervi oggi?",
    "Inglese": "How can I assist you today?",
    "Francese": "Comment puis-je vous aider aujourd’hui ?",
    "Spagnolo": "¿Cómo puedo ayudarle hoy?",
    "Tedesco": "Wie kann ich Ihnen heute helfen?",
    "Russo": "Чем я могу помочь вам сегодня?",
    "Cinese": "我今天能帮你什么？",
    "Arabo": "كيف يمكنني مساعدتك اليوم؟",
    "Portoghese": "Como posso ajudá-lo hoje?",
    "Giapponese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Vectorstore backed retriever",
    "Cohere reranker",
    "Contextual compression",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#                     GitPython integration
####################################################################

def push_vectorstore_to_github(vectorstore_path, commit_message="Updated vectorstore"):
    """
    Pushes the vectorstore directory to a GitHub repository using GitPython.

    Parameters:
        vectorstore_path (str): Path to the vectorstore directory.
        commit_message (str): Commit message for the update.
    """
    try:
        # Initialize the repository
        repo = Repo(vectorstore_path)
        
        # Check for changes
        if repo.is_dirty() or repo.untracked_files:
            # Stage all changes
            repo.git.add(A=True)
            
            # Commit the changes
            repo.index.commit(commit_message)
            
            # Push to the remote repository
            origin = repo.remote(name='origin')
            origin.push()
            print("Vectorstore successfully pushed to GitHub.")
        else:
            print("No changes detected in the vectorstore.")

    except Exception as e:
        print(f"Error pushing vectorstore to GitHub: {e}")

####################################################################
#        Process documents and create vectorstore (Chroma dB)
####################################################################

def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Conversational Retrieval chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
        ):
            error_messages.append(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API"
            )

        if not st.session_state.uploaded_file_list:
            error_messages.append("Seleziona file da caricare")
        if st.session_state.vector_store_name == "":
            error_messages.append("Fornisci un nome per il Vectorstore")

        if len(error_messages) == 1:
            st.session_state.error_message = "Per favore " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Per favore "
                + ", ".join(error_messages[:-1])
                + ", e "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Delete old temp files
                delte_temp_files()

                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        temp_file_path = os.path.join(
                            TMP_DIR.as_posix(), uploaded_file.name
                        )
                        with open(temp_file_path, "wb") as temp_file:
                            temp_file.write(uploaded_file.read())

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)

                    # 5. Embeddings
                    embeddings = select_embeddings_model()

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_directory,
                    )
                    st.info(
                        f"Vectorstore **{st.session_state.vector_store_name}** creato con successo"
                    )

                    # 7. Push vectorstore to GitHub
                    push_vectorstore_to_github(
                        vectorstore_path=persist_directory,
                        commit_message=f"Updated vectorstore: {st.session_state.vector_store_name}"
                    )

            except Exception as error:
                st.error(f"Si è verificato un errore: {str(error)}")

####################################################################
#                       Streamlit Interface
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chatta con il bot")
    with col2:
        st.button("Svuota Chat", on_click=clear_chat_history)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
        ):
            st.info(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API per continuare."
            )
            st.stop()
        with st.spinner("In esecuzione..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
