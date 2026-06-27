import streamlit as st 
import validators
from langchain_classic.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os 
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Universal URL Summarizer & Chat", page_icon="📖", layout="wide")


st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("📖 Universal URL Summarizer & Chat")
st.caption("Paste a YouTube video link or any website URL to instantly get a summary and chat with its content.")
st.divider()

# Sidebar for inputs, Main area for output and chat
with st.sidebar:
    st.header("⚙️ Configuration")
    generic_url = st.text_input("URL", placeholder="Enter YouTube or Website URL...")
    summary_clicked = st.button("⚡ Generate Summary & Load RAG", type="primary")

# llm
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b")

# embeddings 
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# summarization prompt 
prompt_temp = """
Provide a summary of the following content in 1000 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_temp, input_variables=["text"])

# initializing states 
if "store" not in st.session_state:
    st.session_state.store = {}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# Get Summary
def get_url_summary(url):
    if not generic_url:
        st.error("Please enter a valid URL first!")
    else:
        # Added a clean spinner container so users know it's working
        with st.spinner("Processing content and building vector index... This might take a minute."):
            try:
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, language=['hi', 'eng'], add_video_info=False)
                elif not validators.url(url):
                    st.error("Invalid URL format. Please try again.")
                    return
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()
                
                # Storing in vector db
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splites = text_splitter.split_documents(docs)
                vector_store = FAISS.from_documents(documents=splites, embedding=embedding)
                st.session_state.vector_store = vector_store

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                url_summary = chain.run(docs)

                st.session_state.summary = url_summary
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error("Some Error Occurred, TRY AGAIN!")
                st.exception(e)

def convenstional_rag_chain():
    retriever = st.session_state.vector_store.as_retriever()
    
    contextualize_q_system_prompt = (
        """
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        """
    )
    
    contextualize_q__prompt = ChatPromptTemplate.from_messages(
         [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
         ]
    )

    history_aware_retriver = create_history_aware_retriever(llm, retriever, contextualize_q__prompt)

    # Question answer chain 
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriver, question_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id] 
    
    convenstional_rag_chain_obj = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return convenstional_rag_chain_obj


# Execution logic
if summary_clicked:
    get_url_summary(generic_url)

# Display
if st.session_state.summary:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📝 Document Summary")
        # Wrapped inside a nice block container
        st.info(st.session_state.summary)

    with col2:
        st.subheader("💬 Chat with Document")
        

        rag_chain_instance = convenstional_rag_chain()
        
        history_store = st.session_state.store.get("Chat_1")
        if history_store:
            for msg in history_store.messages:
                role = "user" if msg.type == "human" else "assistant"
                with st.chat_message(role):
                    st.write(msg.content)

  
        question = st.chat_input("Ask anything about this URL...")
        if question:

            with st.chat_message("user"):
                st.write(question)
                
           
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain_instance.invoke(  
                        {"input": question},
                        config={"configurable": {"session_id": "Chat_1"}}
                    )
                    st.write(response["answer"])
else:
    
    st.info("← Enter a URL in the sidebar and click 'Generate Summary' to get started.")