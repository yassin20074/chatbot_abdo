import os
import re
import sqlite3
import requests
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from app.schemas import ChatRequest, ChatResponse

load_dotenv(override=False)

app = FastAPI(title="Eyewear Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    return key.strip() if key else None


def clean_llm_response(text: str) -> str:
    """دالة لتنظيف نص الرد من أي رموز مارك داون زائدة أو \\n مكتوبة كـ string"""
    if not text:
        return ""
    
    # 1. استبدال الـ \n المكتوبة كـ Literal String بـ Enter حقيقي
    text = text.replace("\\n", "\n")
    
    # 2. إزالة رموز النجوم الخاصة بالتنسيق البولد (text أو *text*)
    text = re.sub(r"\*{1,2}", "", text)
    
    # 3. إزالة رموز المارك داون الأخرى مثل # أو _ لو وجدت
    text = re.sub(r"[#_`~]", "", text)
    
    # 4. مسح أي مسافات أو أسطر فارغة زائدة في البداية والنهاية
    return text.strip()


def fetch_store_notes() -> str:
    """جلب الملاحظات من الـ Endpoint الخاصة بالباك إند"""
    notes_url = "https://api.hi-vision-optics.com/api/physicallenses/notes-for-chatbot"
    try:
        response = requests.get(notes_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("notes", str(data))
        return "لا توجد ملاحظات إضافية متاحة حالياً."
    except Exception as e:
        print(f"Error fetching notes: {e}")
        return "لا توجد ملاحظات إضافية متاحة حالياً."


# 1. إعداد الـ State الخاص بـ LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: str


# 2. دالة الـ Model
def call_model(state: State):
    notes = state.get("notes", "لا توجد ملاحظات إضافية.")
    api_key = get_groq_api_key()

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # استخدام موديل Llama 3.3 70B على Groq لإعطاء أداء عالي وتنسيق عربي نظيف
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=700
    )

    system_prompt = f"""
أنت مساعد مبيعات خبير واستشاري لموقع متجر نظارات.
مهتك مساعدة العملاء في اختيار النظارة المناسبة لهم بناءً على طلبهم والملاحظات المرفقة للمنتجات.

ملاحظات ونظارات المتجر المتاحة حالياً:
---
{notes}
---

قواعد التفاعل وتنسيق الرد:
1. اكتب بنص عربي سلس وعادي جداً بدون استخدام أي رموز تنسيق مارك داون مثل النجوم (**) أو الهاشتاج (#).
2. لا تدرج رموز إسكيب مثل \\n داخل النص، واكتب الفقرات بشكل طبيعي.
3. استند بشكل أساسي على الملاحظات المرفقة لتوصية النظارات المناسبة.
4. وضح للعميل سبب ترشيح النظارة باختصار.
5. تذكر سياق المحادثة السابق المتاح في الذاكرة للرد بدقة على الأسئلة التابعة.
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
        # جلب الملاحظات تلقائياً من الباك إند
        fetched_notes = fetch_store_notes()

        config = {"configurable": {"thread_id": request.session_id}}

        input_state = {
            "messages": [HumanMessage(content=request.user_prompt)],
            "notes": fetched_notes
        }

        output = app_graph.invoke(input_state, config=config)
        raw_reply = output["messages"][-1].content

        # تنظيف الرد من الرموز والـ \n المكتوبة
        cleaned_reply = clean_llm_response(raw_reply)

        return ChatResponse(reply=cleaned_reply)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Error: {str(e)}"
        )
