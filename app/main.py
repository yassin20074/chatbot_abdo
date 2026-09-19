import os
import sqlite3
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

load_dotenv()

app = FastAPI(title="Eyewear Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1. إعداد الـ State الخاصة بـ LangGraph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: str


# 2. دالة الـ Model
def call_model(state: State):
    notes = state.get("notes", "لا توجد ملاحظات إضافية.")
    api_key = "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"

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.4,
        max_tokens=700
    )

    system_prompt = f"""
أنت مساعد مبيعات خبير واستشاري لموقع متجر نظارات.
مهتك مساعدة العملاء في اختيار النظارة المناسبة لهم بناءً على طلبهم والملاحظات المرفقة للمنتجات.

ملاحظات ونظارات المتجر المتاحة حالياً:
---
{notes}
---

قواعد التفاعل والمساعدة:
1. استند بشكل أساسي ومباشر على النظارات المذكورة في الملاحظات أعلاه لتحديد الخيار الأنسب.
2. وضح للعميل سبب ترشيح النظارة (مثل ملاءمتها لعدسات الحماية، شكل الوجه، الاستخدام المكتبي أو الخارجي).
3. تذكر سياق المحادثة السابق دائماً المتاح في الذاكرة للرد بدقة على استفسارات العميل التابعة (مثل: السعر، اللون، المواصفات).
4. استخدم لغة عربية واضحة، سهلة، ومباشرة بدون إطالة غير ضرورية.
"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# 3. بناء الـ Graph
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# 4. إعداد اتصال حفظ الذاكرة المباشر في قاعدة بيانات SQLite محلية
conn = sqlite3.connect("chat_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app_graph = workflow.compile(checkpointer=memory)


@app.get("/")
def health_check():
    api_key = os.getenv("GROQ_API_KEY")
    return {
        "status": "online",
        "groq_key_found": bool(api_key)
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in server environment variables."
        )

    try:
        # استخدام thread_id لاسترجاع وحفظ ذاكرة session_id الممررة من الفرونت إند
        config = {"configurable": {"thread_id": request.session_id}}

        input_state = {
            "messages": [HumanMessage(content=request.user_prompt)],
            "notes": request.notes or ""
        }

        output = app_graph.invoke(input_state, config=config)
        last_message = output["messages"][-1]

        return ChatResponse(reply=last_message.content)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Error: {str(e)}"
        )
