from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    user_prompt: str

class ChatResponse(BaseModel):
    reply: str
