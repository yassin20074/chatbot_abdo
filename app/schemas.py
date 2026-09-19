from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str          # معرف المحادثة/العميل (عشان الـ MemorySaver يعرف المحادثة)
    user_prompt: str         # سؤال العميل
    notes: Optional[str] = None  # ملاحظات المنتجات/النظارات المتاحة

class ChatResponse(BaseModel):
    reply: str
