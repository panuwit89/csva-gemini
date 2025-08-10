from typing import List, Dict, Optional
from pydantic import BaseModel

class Attachment(BaseModel):
    original_name: str
    mime_type: str
    content_base64: Optional[str] = None
    
class Message(BaseModel):
    role: str
    content: str
    attachments: Optional[List[Attachment]] = None
    
class PromptRequest(BaseModel):
    prompt: str
    conv_id: int
    history: Optional[List[Message]] = None

class ChatRequest(BaseModel):
    conv_id: int
    
class RefreshKnowledgeRequest(BaseModel):
    force: bool = False