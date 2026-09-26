import os
import re
import sqlite3
from typing import Sequence
from typing_extensions import Annotated, TypedDict

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# RAG & Hybrid Search Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.schemas import ChatRequest, ChatResponse

load_dotenv(override=False)

app = FastAPI(title="Eyewear & Company Hybrid RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Hybrid Retriever Variable
hybrid_retriever = None

def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    return key.strip() if key else None


def clean_llm_response(text: str) -> str:
    """دالة لتنظيف نص الرد من أي رموز مارك داون زائدة أو \\n مكتوبة كـ string"""
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = re.sub(r"\*{1,2}", "", text)
    text = re.sub(r"[#_`~]", "", text)
    return text.strip()


def initialize_hybrid_rag(pdf_path: str = "company_info.pdf"):
    """
    تحميل الـ PDF وتجهيز Hybrid Retriever (BM25 + FAISS Dense)
    """
    global hybrid_retriever
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file '{pdf_path}' not found. RAG for company info will be disabled.")
        return None

    try:
        print("Loading PDF for RAG...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)

         # hynbrid rag
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

         
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        print("Hybrid RAG initialized successfully.")
    except Exception as e:
        print(f"Error initializing RAG: {e}")


@app.on_event("startup")
async def startup_event():
 
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(root_dir, "RAG_Abdo.pdf")
    
    print(f"Loading PDF from: {pdf_path}")
    initialize_hybrid_rag(pdf_path)

async def fetch_store_notes() -> str:
    """جلب الملاحظات الخاصة بالنظارات والعدسات"""
    notes_url = "https://api.hi-vision-optics.com/api/physicallenses/notes-for-chatbot"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(notes_url)
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    formatted_notes = []
                    for item in data:
                        if isinstance(item, dict):
                            title = item.get("title") or item.get("name") or ""
                            content = item.get("notes") or item.get("description") or item.get("content") or str(item)
                            formatted_notes.append(f"- {title}: {content}".strip("- :"))
                        else:
                            formatted_notes.append(str(item))
                    return "\n".join(formatted_notes)
                
                elif isinstance(data, dict):
                    if "notes" in data and isinstance(data["notes"], list):
                        return "\n".join([str(n) for n in data["notes"]])
                    return data.get("notes", str(data))
                
                return str(data)
                
        return "لا توجد ملاحظات إضافية متاحة حالياً."
    except Exception as e:
        print(f"Error fetching notes: {e}")
        return "لا توجد ملاحظات إضافية متاحة حالياً."


# 1. إعداد الـ State الخاص بـ LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: str
    rag_context: str


# 2. دالة الـ Model
def call_model(state: State):
    notes = state.get("notes", "لا توجد ملاحظات إضافية.")
    rag_context = state.get("rag_context", "لا تتوفر معلومات إضافية من المستندات.")
    
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=700
    )

    system_prompt = f"""
أنت مساعد الذكاء الاصطناعي الرسمي لموقع ومتجر Hi-Vision Optics.

لديك مصدرين للمعلومات للاستعانة بهما في الرد:

المصدر الأول: ملاحظات العدسات والنظارات (لإجابة أسئلة المنتجات والترشيحات):
---
{notes}
---

المصدر الثاني: معلومات الشركة والمستندات (لإجابة الأسئلة عن الشركة، الخدمات، السياسات، إلخ):
---
{rag_context}
---

قواعد التفاعل وتوجيه الرد:
1. إذا كان سؤال المستخدم عن العدسات، النظارات، وترشيحات المنتجات:
   - استند على "ملاحظات العدسات والنظارات".
   - اذكر: اسم العدسة، والوصف والسبب بوضوح.
2. إذا كان سؤال المستخدم عن الشركة، التعريف بها، السياسات، أو أسئلة عامة عن المؤسسة:
   - استند بشكل مباشر على "معلومات الشركة والمستندات" المرفقة.
3. اكتب بنص عربي سلس ومباشر بدون استخدام أي رموز تنسيق مارك داون غريبة.
4. إذا لم تجد إجابة مباشرة في المصادر، أجب بأقرب معلومة متوفرة بأسلوب لبق.
"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# 3. بناء الـ Graph
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# 4. إعداد الذاكرة عبر SQLite
conn = sqlite3.connect("chat_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app_graph = workflow.compile(checkpointer=memory)


@app.get("/")
def health_check():
    api_key = get_groq_api_key()
    return {
        "status": "online",
        "groq_key_found": bool(api_key),
        "rag_status": "active" if hybrid_retriever is not None else "inactive",
        "key_preview": f"{api_key[:7]}..." if api_key else "NOT_FOUND"
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    api_key = get_groq_api_key()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in server environment variables."
        )

    try:
        # 1. جلب الملاحظات من الـ API
        fetched_notes = await fetch_store_notes()

        # 2. البحث في الـ RAG عبر Hybrid Search للرسالة الحالية
        rag_context = ""
        if hybrid_retriever:
            retrieved_docs = hybrid_retriever.invoke(request.user_prompt)
            rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        config = {"configurable": {"thread_id": request.session_id}}
        input_state = {
            "messages": [HumanMessage(content=request.user_prompt)],
            "notes": fetched_notes,
            "rag_context": rag_context
        }

        output = app_graph.invoke(input_state, config=config)
        raw_reply = output["messages"][-1].content

        # تنظيف الرد
        cleaned_reply = clean_llm_response(raw_reply)

        return ChatResponse(reply=cleaned_reply)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Error: {str(e)}"
        )
