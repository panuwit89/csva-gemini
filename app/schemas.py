from typing import List, Dict, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
    
class PromptRequest(BaseModel):
    prompt: str
    conv_id: int
    history: Optional[List[Message]] = None

class ChatRequest(BaseModel):
    conv_id: int
    
class RefreshKnowledgeRequest(BaseModel):
    force: bool = False