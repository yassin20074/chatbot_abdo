import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

from app.schemas import ChatRequest, ChatResponse
from app.prompt import SYSTEM_PROMPT_TEMPLATE

load_dotenv()

app = FastAPI(
    title="Eyewear Recommendation Chatbot API",
    version="1.0.0"
)

# السماح لربط الـ API بفرونت إند الموقع (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("groq_api")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def health_check():
    return {"status": "online", "service": "Eyewear Chatbot API"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_recommendation(request: ChatRequest):
    try:
        # تجهيز الـ System Prompt وحقن الملاحظات فيه
        notes_content = request.notes if request.notes else "لا يوجد ملاحظات إضافية محدودة حالياً."
        system_content = SYSTEM_PROMPT_TEMPLATE.format(notes=notes_content)

        # تجهيز الرسائل للـ LLM
        messages = [{"role": "system", "content": system_content}]

        # إضافة السجل السابق إن وجد
        if request.chat_history:
            messages.extend(request.chat_history)

        # إضافة سؤال العميل الحالي
        messages.append({"role": "user", "content": request.user_prompt})

        # الاستدعاء لـ Groq API (استخدام llama-3.3-70b-versatile أو gemma2-9b-it)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=600,
        )

        bot_reply = completion.choices[0].message.content
        return ChatResponse(reply=bot_reply)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )