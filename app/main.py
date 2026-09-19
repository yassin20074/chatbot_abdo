 import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.schemas import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="Eyewear Chatbot API with LangGraph Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. تهيئة LLM عبر Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.4,
    max_tokens=700
) if GROQ_API_KEY else None


# 2. تعريف حالة الـ Graph (State)
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: str


# 3. دالة الـ Node الخاصة بالـ LLM
def call_model(state: State):
    notes = state.get("notes", "لا توجد ملاحظات إضافية.")
    
    system_prompt = f"""
أنت مساعد مبيعات خبير واستشاري لموقع متجر نظارات.
همتك مساعدة العملاء في اختيار النظارة المناسبة لهم بناءً على طلبهم والملاحظات المرفقة للمنتجات.

ملاحظات ونظارات المتجر المتاحة حالياً:
---
{notes}
---

قواعد التفاعل والمساعدة:
1. استند بشكل أساسي ومباشر على النظارات المذكورة في الملاحظات أعلاه لتحديد الخيار الأنسب.
2. وضح للعميل سبب ترشيح النظارة (مثل ملاءمتها لعدسات الحماية، شكل الوجه، الاستخدام المكتبي أو الخارجي).
3. تذكر سياق المحادثة السابق دائماً عند الإجابة عن التفاصيل مثل (السعر، اللون، أو الميزات).
4. استخدم لغة عربية واضحة، سهلة، ومباشرة بدون إطالة غير ضرورية.
"""
    
    # دمج الـ System Prompt مع تاريخ الرسائل
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# 4. بناء الـ LangGraph
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# 5. استخدام MemorySaver لحفظ الذاكرة تلقائياً
checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)


@app.get("/")
def health_check():
    return {
        "status": "online",
        "groq_configured": bool(GROQ_API_KEY)
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not GROQ_API_KEY or not llm:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in server environment variables."
        )

    try:
        # إعداد الـ config لتحديد الـ thread_id (لكل session_id ذاكرة مستقلة)
        config = {"configurable": {"thread_id": request.session_id}}

        # إرسال الرسالة والملاحظات للـ Graph
        input_state = {
            "messages": [HumanMessage(content=request.user_prompt)],
            "notes": request.notes or ""
        }

        # تشغيل الـ Graph
        output = app_graph.invoke(input_state, config=config)

        # أخذ آخر رسالة من الـ AI
        last_message = output["messages"][-1]
        
        return ChatResponse(reply=last_message.content)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Error: {str(e)}"
        )
