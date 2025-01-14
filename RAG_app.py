####################################################################
#                         import
####################################################################
import os
os.environ["PATH"] = os.path.join(os.path.dirname(__file__), "bin") + ":" + os.environ["PATH"]

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import glob
from pathlib import Path
import sqlite3
print("SQLite version:", sqlite3.sqlite_version)

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

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# # HuggingFace
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_community.llms import HuggingFaceHub

# Import streamlit
import streamlit as st

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
  #  ":hugging_face: **HuggingFace**",
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
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Chat With Your Data")

st.title("🤖 Chatbot")

# API keys
st.session_state.openai_api_key = ""
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""
st.session_state.hf_api_key = ""


def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            value=token1,
            placeholder="insert your API key",
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            value=token2,
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.hf_api_key = ""

    # if LLM_provider == "HuggingFace":
    #     st.session_state.hf_api_key = st.text_input(
    #         text_input_API_key,
    #         type="password",
    #         placeholder="insert your API key",
    #     )
    #     st.session_state.openai_api_key = ""
    #     st.session_state.google_api_key = ""

    with st.expander("**Modelli e parametri**"):
        st.session_state.selected_model = st.selectbox(
            f"Scegli modello {LLM_provider}", list_models
        )

        # model parameters
        st.session_state.temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )


def sidebar_and_documentChooser():
    """Create the sidebar and a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an existing vectorstore)."""

    with st.sidebar:
        st.caption(
            "🚀 Un chatbot basato su Retrieval Augmented Generation che utilizza 🔗 Langchain, Cohere, OpenAI e Google Generative AI"
        )
        st.write("")

        llm_chooser = st.radio(
            "Seleziona provider",
            list_LLM_providers,
            captions=[
                "[Opzioni di acquisto](https://openai.com/pricing)",
                "Limite: 60 richieste al minuto",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider="OpenAI",
                text_input_API_key="Inserisci chiave OpenAi - [OpenAI API Key](https://platform.openai.com/account/api-keys)",
                list_models=[
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo-preview",
                ],
            )

        if llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                LLM_provider="Google",
                text_input_API_key="Inserisci chiave Google API - [Google API Key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro"],
            )

        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Lingua dell'assistente", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retriever")
        retrievers = list_retriever_types
        if st.session_state.selected_model == "gpt-3.5-turbo":
            retrievers = list_retriever_types[:-1]

        st.session_state.retriever_type = st.selectbox(
            f"Seleziona tipo di retriever", retrievers
        )
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[1]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Inserisci chiave Cohere API - [Cohere API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                value=token3,
                placeholder="Inserisci chiave Cohere API",
            )

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore
    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Crea un nuovo Vectorstore", "Seleziona un Vectorstore esistente"]
    )

    with tab_new_vectorstore:
        # 1. Select documents
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Seleziona documenti**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**I documenti saranno caricati e salvati in un vectorstore (Chroma dB). Inserisci un nome valido**",
            placeholder="Nome Vectorstore",
        )
        # 3. Add a button to process documents and create a Chroma vectorstore
        st.button("Crea Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        # Replace tkinter file dialog with Streamlit input for directory name
        st.write("Seleziona un Vectorstore:")

        # Text input for vectorstore path
        st.session_state.selected_vectorstore_name = st.text_input(
            "Inserisci il percorso del Vectorstore",
            placeholder="Inserisci il percorso o nome del Vectorstore salvato",
        )

        if st.button("Carica Vectorstore"):
            # Check inputs
            error_messages = []
            if (
                not st.session_state.openai_api_key
                and not st.session_state.google_api_key
                and not st.session_state.hf_api_key
            ):
                error_messages.append(
                    f"Inserisci la tua chiave {st.session_state.LLM_provider} API"
                )

            if (
                st.session_state.retriever_type == list_retriever_types[0]
                and not st.session_state.cohere_api_key
            ):
                error_messages.append(f"Inserisci la tua chiave Cohere API")

            if len(error_messages) == 1:
                st.session_state.error_message = "Per favore " + error_messages[0] + "."
                st.warning(st.session_state.error_message)
            elif len(error_messages) > 1:
                st.session_state.error_message = (
                    "Per favore "
                    + ", ".join(error_messages[:-1])
                    + ", e "
                    + error_messages[-1]
                    + "."
                )
                st.warning(st.session_state.error_message)

            # Load Vectorstore if inputs are valid
            elif st.session_state.selected_vectorstore_name:
                try:
                    with st.spinner("Caricamento vectorstore..."):
                        # Load Chroma vectorstore
                        embeddings = select_embeddings_model()
                        st.session_state.vector_store = Chroma(
                                embedding_function=embeddings,
                                persist_directory=selected_vectorstore_path,
                            )
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_vectorstore_path,
                        )

                        # Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # Clear chat history
                        clear_chat_history()

                        st.info(
                            f"**{st.session_state.selected_vectorstore_name}** è stato caricato con successo"
                        )
                except Exception as e:
                    st.error(e)
            else:
                st.warning("Per favore inserisci un nome valido per il Vectorstore.")

####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################
def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    """
    Crete documnet loaders for PDF, TXT and CSV files.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Select embeddings models: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings."""
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    if st.session_state.LLM_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=st.session_state.google_api_key
        )

    # if st.session_state.LLM_provider == "HuggingFace":
    #     embeddings = HuggingFaceInferenceAPIEmbeddings(
    #         api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large"
    #     )

    return embeddings


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

        cohere_api_key: Cohere API key
        cohere_model (str): model used by Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"]
        cohere_top_n: top n documents returned bu Cohere, default = 10

    """

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever: a Vectorstore-backed retriever.
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        k (int): top k relevant documents to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents

    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    """Build a ContextualCompressionRetriever using CohereRerank endpoint to reorder the results
    based on relevance to the query.

    Parameters:
       base_retriever: a Vectorstore-backed retriever
       cohere_api_key: the Cohere API key
       cohere_model: the Cohere model, in ["rerank-multilingual-v2.0","rerank-english-v2.0"], default = "rerank-multilingual-v2.0"
       top_n: top n results returned by Cohere rerank. default = 10.
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere


def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            error_messages.append(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API"
            )

        if (
            st.session_state.retriever_type == list_retriever_types[0]
            and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"Inserisci la tua chiave Cohere API")
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
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errori: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()
                    print(f"Caricato/i {len(documents)} documento/i")
                    for doc in documents:
                        print(doc.metadata, doc.page_content[:100])  # Print metadata and first 100 chars

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    print(f"Creati {len(chunks)} chunks")
                    # 5. Embeddings
                    embeddings = select_embeddings_model()
                    print("Modello di embedding creato con successo")
                    test_vector = embeddings.embed_query("Test embedding")
                    print(f"Generato test embedding: {test_vector[:10]}")  # First 10 dimensions

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** creato con successo"
                        )

                        # 7. Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Cclear chat_history
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"Si è verificato un errore: {str(error)}")


####################################################################
#                       Create memory
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models"""

    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024  # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="italian"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="italian",
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases
    the question and generates a standalone query.
    This query is then sent to the retriever, which fetches relevant documents (context)
    and passes them along with the standalone question and chat history to an LLM to answer.
    """

    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory(st.session_state.selected_model)

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    if st.session_state.LLM_provider == "OpenAI":
        standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )
    if st.session_state.LLM_provider == "Google":
        standalone_query_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        response_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            convert_system_message_to_human=True,
        )

    # if st.session_state.LLM_provider == "HuggingFace":
    #     standalone_query_generation_llm = HuggingFaceHub(
    #         repo_id=st.session_state.selected_model,
    #         huggingfacehub_api_token=st.session_state.hf_api_key,
    #         model_kwargs={
    #             "temperature": 0.1,
    #             "top_p": 0.95,
    #             "do_sample": True,
    #             "max_new_tokens": 1024,
    #         },
    #     )
    #     response_generation_llm = HuggingFaceHub(
    #         repo_id=st.session_state.selected_model,
    #         huggingfacehub_api_token=st.session_state.hf_api_key,
    #         model_kwargs={
    #             "temperature": st.session_state.temperature,
    #             "top_p": st.session_state.top_p,
    #             "do_sample": True,
    #             "max_new_tokens": 1024,
    #         },
    #     )

    # 5. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # if st.session_state.LLM_provider == "HuggingFace":
        #     answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Documenti di riferimento**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Pagina: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Fonte: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)


####################################################################
#                         Chatbot
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
            and not st.session_state.hf_api_key
        ):
            st.info(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API per continuare."
            )
            st.stop()
        with st.spinner("In esecuzione..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
