# Environment
from dotenv import load_dotenv
import os

load_dotenv()

# Standard library
import re
import logging

# YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from pytubefix import YouTube

# LangChain — LLMs & Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# LangChain — Splitters
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain — Vector Store & Retrievers
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Reranker
from sentence_transformers import CrossEncoder

# Chat History
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Enable logging to debug generated queries
logging.basicConfig(level=logging.INFO)
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# ---------------------------------------------------------------------------
# Global Prompts
# ---------------------------------------------------------------------------


prompt = PromptTemplate(
    template="""
    You are a helpful assistant that answers questions based on a YouTube video transcript.

    Here are the most relevant excerpts from the transcript:

    <context>
    {context}
    </context>

    Instructions:
    - First identify which parts of the context are relevant to the question
    - Answer based ONLY on the provided context
    - If the context doesn't contain enough information, say exactly: "I don't have enough information in the transcript to answer this." Then explain in 2-3 sentences what specific information was missing.
    - Be concise and clear
    - If the answer has multiple parts, use bullet points

    Question: {question}

    Answer:
    """,
    input_variables=['context', 'question']
)

condense_prompt = PromptTemplate(
    template="""You are a question rewriter for a RAG (Retrieval-Augmented Generation) system.
Your job is to rewrite follow-up questions into standalone questions that are
optimized for searching a vector store and keyword retrieval.

Rules:
- Replace all pronouns and vague references with their specific entities from the chat history
- Make the question keyword-rich and specific for document retrieval
- Do NOT answer the question — only rewrite it
- If the question already makes sense on its own, return it unchanged

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
    input_variables=["chat_history", "question"]
)


# ---------------------------------------------------------------------------
# Condense chain (video-independent — safe to initialise globally)
# ---------------------------------------------------------------------------
condense_chain = condense_prompt | llm | StrOutputParser()




def clean_transcript(text):
    # Remove filler words
    fillers = r'\b(um|uh|like|you know|basically|actually|literally|so|right)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    
    # Fix multiple spaces left behind
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def rerank_docs(input_dict, top_k=5):
    question = input_dict["question"]
    docs = input_dict["multi_query_context"]
    
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    return [doc for _, doc in scored_docs[:top_k]]


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

index_cache = {}

def build_index(video_id):

    if(video_id in index_cache):
        return index_cache[video_id]

    #getting transcript
    try:
        api = YouTubeTranscriptApi()  # create instance
        transcript_list = api.fetch(video_id, languages=["en"])

        transcript = " ".join(chunk.text for chunk in transcript_list)
        print(transcript)
    except TranscriptsDisabled:
        print("No captions available for this video.")
    #cleaning
    transcript = clean_transcript(transcript)
    
    #chunking
    # Pass 1: Semantic chunking — cuts where meaning shifts most
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )

    semantic_chunks = splitter.create_documents([transcript])

    # Pass 2: Enforce a max size so no chunk is too large
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    final_chunks = char_splitter.split_documents(semantic_chunks)
     
     #Add Metadata
    for i, chunk in enumerate(final_chunks):
        chunk.metadata = {"chunk_index": i, "source": f"youtube_{video_id}"}
    print(f"Total chunks: {len(final_chunks)}")

    #creating vector store
    vector_store = FAISS.from_documents(final_chunks, embeddings)

    # BM25 — keyword-based retriever (inverted index over chunked docs)
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    bm25_retriever.k = 3

    # FAISS — semantic retriever
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Combine — weights must sum to 1.0 (0.4 keyword, 0.6 semantic)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )
    # Stage 1: retrieve via multi-query retriever
    multi_query_chain = RunnableParallel({
        "multi_query_context": RunnableLambda(lambda x: x["question"]) | multi_query_retriever,
        "question": RunnableLambda(lambda x: x["question"])  # extract string from dict
    })

    # Stage 2: Rerank results → format into context string
    parallel_chain = RunnableParallel({
        "context": multi_query_chain | RunnableLambda(rerank_docs) | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"])  # extract string from dict
    })

    # Stage 3: Feed context + question into prompt → LLM → parse output
    main_chain = parallel_chain | prompt | llm | StrOutputParser()

    index_cache[video_id] = {
        "main_chain": main_chain,
        "chat_history": ChatMessageHistory()
    }
    return index_cache[video_id]


def format_chat_history(chat_history, k=5):
    """Format the last k turns of chat history into a readable string."""
    messages = chat_history.messages[-(k * 2):]
    
    if not messages:
        return ""
    
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    
    return "\n".join(formatted)


def ask_question(video_id , question):
    if(video_id not in index_cache):
        build_index(video_id)
    main_chain = index_cache[video_id]["main_chain"]
    chat_history = index_cache[video_id]["chat_history"]
    history_str = format_chat_history(chat_history)
    standalone_q = question 
    if(history_str): standalone_q = condense_chain.invoke({"question" : question , "chat_history" : history_str})
    answer = main_chain.invoke({"question" : standalone_q})
    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)
    return answer

