import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

from app.schemas import ChatRequest, ChatResponse
from app.prompt import SYSTEM_PROMPT_TEMPLATE

load_dotenv()

app = FastAPI(title="Eyewear Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# جلب المفتاح بدون إلقاء RuntimeError عند التشغيل الابتدائي
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# إنابة الـ Client فقط لو المفتاح موجود
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.get("/")
def health_check():
    return {
        "status": "online",
        "groq_configured": bool(GROQ_API_KEY)
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # التحقق وقت طلب الـ API فقط
    if not client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in server environment variables.",
        )

    try:
        notes_text = (
            request.notes
            if request.notes
            else "لا يوجد ملاحظات أو منتجات معينة ممررة."
        )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(notes=notes_text)

        messages = [{"role": "system", "content": system_prompt}]

        if request.chat_history:
            messages.extend(request.chat_history)

        messages.append({"role": "user", "content": request.user_prompt})

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=700,
        )

        return ChatResponse(reply=completion.choices[0].message.content)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API Error: {str(e)}",
        )
