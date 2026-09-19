from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    user_prompt: str
    notes: Optional[str] = None  # ملاحظات المنتجات المتوفرة أو تفاصيل النظارات
    chat_history: Optional[List[Dict[str, str]]] = []  # لسياق المحادثة السابق

class ChatResponse(BaseModel):
    reply: str