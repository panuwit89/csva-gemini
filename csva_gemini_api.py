from typing import List, Dict, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
import re
import os
import uuid
import tempfile
import unicodedata
import uvicorn
import requests
import json
import threading
import datetime
import gc  # เพิ่ม garbage collector

# System instruction
SYSTEM_INSTRUCTION = """วัตถุประสงค์และเป้าหมาย: 
* ตอบคำถามของนิสิตภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเกษตรศาสตร์ เกี่ยวกับเรื่องต่างๆ ที่เกี่ยวข้องกับการศึกษาในภาควิชา ในมหาวิทยาลัยหรือตามความเหมาะสม.
* ให้ข้อมูลที่ถูกต้องและเป็นปัจจุบัน อ้างอิงจากไฟล์ความรู้ที่แนบมา.
* ช่วยเหลือและแนะนำนิสิตในเรื่องต่างๆ เช่น เงื่อนไขการให้ทุนการศึกษาภาคพิเศษ, ขั้นตอนการยื่นหนังสือสหกิจศึกษา, การขอแบบฟอร์มเอกสารต่างๆ, การส่งโครงงานวิทยาการคอมพิวเตอร์, เอกสารที่จำเป็นสำหรับการยื่นสำเร็จการศึกษา, ปฏิทินการศึกษาของภาคเรียน, ขั้นตอนการขอยืมอุปกรณ์ IOT, และขั้นตอนการขอเอกสารทางการศึกษาจากมหาวิทยาลัย.
* สร้างบรรยากาศการสนทนาที่เป็นกันเองและเข้าถึงง่าย.
พฤติกรรมและข้อกำหนด:
1) การตอบคำถามเริ่มต้น:
    ก) ทักทายและแนะนำตัวเองในฐานะเจ้าหน้าที่เสมือนของภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเกษตรศาสตร์.
    ข) สอบถามนิสิตเกี่ยวกับคำถามหรือหัวข้อที่ต้องการทราบ.
    ค) หากนิสิตไม่ได้ระบุหัวข้อที่ชัดเจน ให้สอบถามเพื่อทำความเข้าใจความต้องการของนิสิต.
2) การให้ข้อมูล:
    ก) ให้ข้อมูลที่ถูกต้อง ครบถ้วน และอ้างอิงจากไฟล์ความรู้ที่แนบมา.
    ข) อธิบายข้อมูลด้วยภาษาที่เข้าใจง่าย ชัดเจน และเป็นกันเอง.
    ค) หากจำเป็น สามารถให้ข้อมูลเพิ่มเติมหรือยกตัวอย่างเพื่อให้นิสิตเข้าใจได้ดียิ่งขึ้น.
    ง) หลีกเลี่ยงการใช้ศัพท์เทคนิคที่ซับซ้อนโดยไม่จำเป็น หรืออธิบายศัพท์เทคนิคเหล่านั้นให้เข้าใจง่าย.
3) การช่วยเหลือและการแนะนำ:
    ก) ให้คำแนะนำและแนวทางที่ชัดเจนแก่นิสิตในเรื่องต่างๆ ที่เกี่ยวข้อง.
    ข) อธิบายขั้นตอนและเอกสารที่จำเป็นสำหรับการดำเนินการต่างๆ อย่างละเอียด.
    ค) หากไม่สามารถตอบคำถามได้โดยตรงจากไฟล์ความรู้ ให้แจ้งนิสิตอย่างสุภาพและแนะนำแหล่งข้อมูลอื่นที่อาจเป็นประโยชน์.
4) น้ำเสียงและลักษณะท่าทาง:
    ก) ใช้ภาษาที่เป็นกันเอง สุภาพ และให้เกียรตินิสิต.
    ข) แสดงความกระตือรือร้นที่จะช่วยเหลือและตอบคำถามของนิสิต.
    ค) ตอบคำถามด้วยความอดทนและใจเย็น.
การอ้างอิง:
* ยึดมั่นในการให้ข้อมูลที่ถูกต้องตามที่ระบุไว้ในไฟล์ความรู้ที่แนบมาเท่านั้น.
* นอกเหนือจากนี้ ให้แจ้งว่าไม่สามารถให้ข้อมูลในส่วนนั้นได้โดยตรง แต่สามารถให้ข้อมูลอื่นที่เกี่ยวข้องได้."""

# Transcript instruction
TRANSCRIPT_INSTRUCTION = """
1) บทบาทและบริบท
    * คุณเป็นผู้ช่วยประจำภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเกษตรศาสตร์
    * หน้าที่หลักคือช่วยตอบคำถามเกี่ยวกับทรานสคริปต์ผลการเรียนของนิสิต

รูปแบบการตอบคำถาม:
2) จัดเรียงผลการเรียนตามภาคการศึกษา (“Semester”) และปีการศึกษา ดังนี้:
    - Summer Semester (ปีการศึกษา)
    - First Semester (ปีการศึกษา)
    - Second Semester (ปีการศึกษา)
    * เรียงปีการศึกษาจากน้อยไปมาก
    * ภายในแต่ละปีการศึกษา ให้เรียงลำดับภาคการศึกษาตามที่กำหนด (summer → first → second)
    * ภาคการศึกษา summer อาจจะมีหรือไม่มีขึ้นอยู่กับข้อมูลที่ได้รับมา

3) โครงสร้างการแสดงผล
    * แต่ละภาคการศึกษาขึ้นหัวข้อด้วยชื่อภาคการศึกษาและปี แสดงภาคและปีการศึกษาในรูปแบบตัวหนา เช่น 
        - First Semester (2022)
        - Second Semester (2022)
        - Summer Semester (2023)
        - First Semester (2023)
        - Second Semester (2023)
        - Summer Semester (2024)
        - First Semester (2024)
        - Second Semester (2024)
        - Summer Semester (2025)

4) การตรวจสอบความถูกต้อง
    * หลังแสดงข้อมูลทรานสคริปต์ทั้งหมดแล้ว ให้ถามยืนยันกับเจ้าของข้อมูลเสมอ เช่น
        - “ข้อมูลทรานสคริปต์ที่แสดงข้างต้นถูกต้องหรือไม่ หรือต้องการแก้ไขเพิ่มเติมในส่วนใดบ้างครับ/ค่ะ?”

5) ข้อควรระวัง
    * ควรอ่านไฟล์ทรานสคริปต์และข้อมูลดิบ (input file) ให้ละเอียดและครบถ้วนก่อนจัดเรียง
    * หากเจอข้อมูลขาดหายหรือไม่ชัดเจน ให้สอบถามเพิ่มเติมทันที"""

# Base URL for Laravel API
LARAVEL_BASE_URL = "http://localhost"

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

# Initialize the Gemini client with provided API key
client = genai.Client(api_key="AIzaSyB6AS0OkeKNbGecKzVIidL4vXsvUa-OgVo")

# Dictionary to store chat sessions
chat_sessions: Dict[int, object] = {}

# Global variable to store processed knowledge files
knowledge_contents = []

# Collect refresh status information
refresh_status = {
    "is_running": False,
    "last_refresh": None,
    "files_processed": 0,
    "error": None,
    "start_time": None,
    "end_time": None
}

# Lock สำหรับ thread safety
refresh_lock = threading.Lock()

def fetch_active_knowledge_files():
    """Fetch active knowledge files from Laravel API"""
    try:
        response = requests.get(f"{LARAVEL_BASE_URL}/knowledge/active", timeout=30)
        response.raise_for_status()
        
        knowledge_data = response.json()
        print(f"Fetched {len(knowledge_data)} active knowledge files from Laravel")
        
        return knowledge_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching knowledge files from Laravel: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from Laravel: {e}")
        return []

def download_file_from_laravel(file_path):
    """Download file content from Laravel storage"""
    try:
        file_url = f"{LARAVEL_BASE_URL}/storage/{file_path}"
        response = requests.get(file_url, timeout=60)
        response.raise_for_status()
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file {file_path}: {e}")
        return None

def process_knowledge_files_from_laravel():
    """Process knowledge files from Laravel API"""
    knowledge_files = fetch_active_knowledge_files()
    
    if not knowledge_files:
        print("No active knowledge files found")
        return []
    
    contents = []
    
    for knowledge in knowledge_files:
        try:
            file_path = knowledge.get('file_path')
            filename = knowledge.get('filename')
            title = knowledge.get('title', 'Unknown')
            
            if not file_path or not filename:
                print(f"Missing file path or filename for knowledge: {title}")
                continue
            
            # Download file content
            file_content = download_file_from_laravel(file_path)
            
            if file_content is None:
                print(f"Failed to download file: {filename}")
                continue
            
            # Create temporary file for upload
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, filename)
            
            try:
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_content)
                
                # Upload to Gemini
                uploaded_file = client.files.upload(
                    file=temp_file_path,
                    config=dict(mime_type='application/pdf')
                )
                contents.append(uploaded_file)
                
                print(f"Processed: {title} ({filename})")
                
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")
            
        except Exception as e:
            print(f"Error processing knowledge file {knowledge.get('title', 'Unknown')}: {e}")
            continue
    
    print(f"Successfully processed {len(contents)} knowledge files")
    return contents

def refresh_knowledge_base():
    """Refresh knowledge base from Laravel - Fixed version"""
    global knowledge_contents
    
    with refresh_lock:  # Thread safety
        if refresh_status["is_running"]:
            print("Refresh already in progress, skipping...")
            return len(knowledge_contents)
        
        refresh_status["is_running"] = True
        refresh_status["start_time"] = datetime.datetime.now()
        refresh_status["error"] = None
    
    try:
        print("Refreshing knowledge base from Laravel...")
        
        # Clear old knowledge contents to free memory
        old_contents = knowledge_contents
        knowledge_contents = []
        
        # Force garbage collection
        del old_contents
        gc.collect()
        
        # Process new knowledge files
        new_contents = process_knowledge_files_from_laravel()
        knowledge_contents = new_contents  # Update global variable
        
        # Edit: not update existing chat sessions
        print("Knowledge base refreshed successfully")
        print("Note: New chat sessions will use updated knowledge. Existing chat session contents remain unchanged.")
        
        refresh_status["files_processed"] = len(knowledge_contents)
        refresh_status["last_refresh"] = datetime.datetime.now().isoformat()
        
        return len(knowledge_contents)
        
    except Exception as e:
        error_msg = str(e)
        refresh_status["error"] = error_msg
        print(f"Error refreshing knowledge base: {error_msg}")
        raise e
        
    finally:
        refresh_status["is_running"] = False
        refresh_status["end_time"] = datetime.datetime.now()
        # Force garbage collection after refresh
        gc.collect()

def create_chat_session(conv_id: int, history: list[types.Content] | None = None):
    """Create a new chat session with the Gemini model and configuration"""
    try:
        # If this is a new conversation ID, initialize an empty chat history
        if history is None:
            history = []
            
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
            history=history,
        )
        chat_sessions[conv_id] = chat
        print(f"Chat session {conv_id} created successfully.")
        return chat
    except Exception as e:
        print(f"Error creating chat session {conv_id}: {e}")
        raise e

def get_chat_session(conv_id: int, history: list[types.Content] | None = None):
    """Get existing chat by conversation ID.
    If not found in memory, it will try to fetch history from the Laravel backend
    and recreate the session.
    """
    if conv_id in chat_sessions:
        print(f"Found chat session {conv_id} in memory.")
        return chat_sessions[conv_id]
    
    if conv_id not in chat_sessions:
        print(f"Chat session {conv_id} not in memory. Creating new session...")
        
    try:
        recreated_history = []
        if history is not None:
            for i, msg in enumerate(history):
                
                role_val = None
                content_val = None

                if isinstance(msg, Message): # Is Pydantic Message object type (from sendPrompt)
                    role_val = msg.role
                    content_val = msg.content
                elif isinstance(msg, dict): # Is dictionary type (from process_files_and_prompt)
                    role_val = msg.get('role', msg.get('type', 'user'))
                    content_val = msg.get('content', '')
                else:
                    # Handle unexpected types
                    print(f"WARNING: Unexpected type in history item {i}: {type(msg)}. Attempting to convert to string content.")
                    role_val = "user" # Initialize role as 'user' if not specified
                    content_val = str(msg) # Convert to string content to avoid AttributeError

                if role_val is not None and content_val is not None:
                    recreated_history.append(
                        types.Content(parts=[types.Part(text=content_val)], role=role_val)
                    )
                else:
                    print(f"WARNING: Skipping malformed or empty history message for conv_id {conv_id}: {msg}")

        print(f"Successfully fetched {len(recreated_history)} messages. Recreating session...")
        
        chat = create_chat_session(conv_id, history=recreated_history)
        
        try:
            initialize_chat_with_docs(chat)
            print(f"Chat session {conv_id} initialized with knowledge documents")
        except Exception as doc_error:
            print(f"Warning: Could not initialize chat {conv_id} with documents: {doc_error}")
        
        return chat

    except requests.exceptions.RequestException as e:
        # Handle API call failures (e.g., network error, 404 not found from Laravel)
        print(f"Failed to fetch history for conv_id {conv_id} from API: {e}")
        raise ValueError(f"Chat session '{conv_id}' not found in memory or database.") from e
    except Exception as e:
        print(f"An unexpected error occurred while recreating session {conv_id}: {e}")
        raise e

def process_prompt(prompt, conv_id, history=None):
    """Process a text prompt"""
    try:
        # Get the chat session for the conversation ID
        chat = get_chat_session(conv_id, history)
        
        # Send the prompt to Gemini and get the response
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Error processing prompt for conv_id {conv_id}: {e}")
        return f"Error: {str(e)}"

def process_files_and_prompt(files, custom_prompt, conv_id, custom_config, history=None):
    """Process uploaded files and a prompt"""
    try:
        # Get the chat session for the conversation ID
        chat = get_chat_session(conv_id, history)

        # List to store uploaded files (Gemini File objects)
        uploaded_gemini_files = []

        # Process each uploaded file
        for file in files:
            # Determine file type based on extension
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == '.pdf':
                mime_type = 'application/pdf'
            elif file_extension == '.txt':
                mime_type = 'text/plain'
            else:
                return f"Unsupported file type: {file_extension}"
            
            # Upload file to Gemini by passing the file path directly
            uploaded_file_obj = client.files.upload(
                file=file.name,
                config=dict(mime_type=mime_type)
            )
            uploaded_gemini_files.append(uploaded_file_obj)
        
        if not uploaded_gemini_files:
            return "No files were uploaded. Please upload at least one file."
        
        chat.send_message(uploaded_gemini_files)
        response = chat.send_message(custom_prompt, custom_config)
        return response.text
        
    except Exception as e:
        print(f"Error processing files and prompt for conv_id {conv_id}: {e}")
        return f"Error: {str(e)}"

def refresh_knowledge_background():
    """Background task for refreshing knowledge base"""
    try:
        print("Starting background knowledge refresh...")
        count = refresh_knowledge_base()
        print(f"Background knowledge refresh completed. Files processed: {count}")
    except Exception as e:
        print(f"Error in background knowledge refresh: {e}")

def initialize_chat_with_docs(chat):
    """Initialize a chat session with documents"""
    try:
        if knowledge_contents:
            chat.send_message(knowledge_contents)
            print("Chat initialized with knowledge documents")
        else:
            print("No knowledge contents available for initialization")
    except Exception as e:
        print(f"Warning: Could not initialize chat with documents: {e}")

def define_chat_name(conv_id: int):
    """Define a name for the chat session based on the first actual user interaction"""
    try:
        if conv_id not in chat_sessions:
            print(f"Error: Chat session '{conv_id}' not found.")
            return "New Conversation"

        chat = get_chat_session(conv_id)
        
        ### Section 1 : Find the first actual user interaction ###
        start_index = -1
        for i, message in enumerate(chat.get_history()):
            # Check if the message is from the user and has text parts
            if message.role == 'user' and any(hasattr(part, 'text') and part.text and part.text.strip() for part in message.parts):
                start_index = i
                break

        # If cannot find a user message, return a default name
        if start_index == -1:
            print(f"No actual user conversation found in conv_id '{conv_id}' to generate a name.")
            return "New Conversation"

        ### Section 2 : Get history content from start_index ###
        # Get the last 4 messages starting from start_index
        history_slice = chat.get_history()[start_index : start_index + 4]
        
        # Create a transcript from history_slice
        transcript_parts = []
        for message in history_slice:
            # Collect all text parts of the same message and check if part.text is not None
            full_text = " ".join([part.text for part in message.parts if hasattr(part, 'text') and part.text is not None]).strip()
            if not full_text:
                continue

            role = "ผู้ใช้" if message.role == "user" else "ผู้ช่วย"
            transcript_parts.append(f"{role}: {full_text}")
        
        # If no messages were found or transcript_parts is empty, return a default name
        if not transcript_parts:
            print(f"No valid transcript found for conv_id '{conv_id}'")
            return "New Conversation"

        transcript = "\n".join(transcript_parts)
        print(f"Transcript for naming conv_id '{conv_id}': {transcript}")
        
        # Prompt for naming the chat based on the transcript content
        prompt_for_naming = f"""
        จากบทสนทนาต่อไปนี้ โปรดตั้งชื่อเรื่องเป็นภาษาไทยที่สั้น กระชับ และได้ใจความ
        โดยมีความยาวไม่เกิน 8 คำ และไม่ต้องใส่เครื่องหมายใดๆ รวมถึงเครื่องหมายคำพูด ("") หรือคำนำหน้าใดๆ เช่น "ชื่อเรื่อง:" 

        บทสนทนา:
        ---
        {transcript}
        ---

        ชื่อเรื่องที่เหมาะสมคือ:
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_for_naming,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            # Check if response is valid and has text
            if response and hasattr(response, 'text') and response.text:
                chat_name = response.text.strip().strip('"').strip("'")
                print(f"Generated name for conv_id '{conv_id}': {chat_name}")
                return chat_name
            else:
                print(f"Gemini API returned empty response for conv_id '{conv_id}'")
                print(f"Response object: {response}")
                if hasattr(response, 'candidates') and response.candidates:
                    print(f"Candidates: {response.candidates}")
                return "New Conversation"
                
        except Exception as gemini_error:
            print(f"Error calling Gemini API for conv_id '{conv_id}': {gemini_error}")
            return "New Conversation"

    except Exception as e:
        print(f"An error occurred while defining chat name for conv_id {conv_id}: {e}")
        return "New Conversation"



# Initialize knowledge base on startup
print("Initializing knowledge base from Laravel...")
try:
    knowledge_contents = process_knowledge_files_from_laravel()
    print(f"Startup: Loaded {len(knowledge_contents)} knowledge files")
except Exception as e:
    print(f"Error during startup knowledge loading: {e}")
    knowledge_contents = []

# Create FastAPI app for API endpoints
app = FastAPI()

@app.post("/api/create_chat")
async def create_chat_api(request: ChatRequest, background_tasks: BackgroundTasks):
    """API endpoint for creating a new chat session"""
    try:
        if request.conv_id in chat_sessions:
            return {"error": f"Chat session {request.conv_id} already exists"}
        
        chat = create_chat_session(request.conv_id)
        
        background_tasks.add_task(initialize_chat_with_docs, chat)
        
        return {"message": f"Chat session {request.conv_id} created successfully"}
    except Exception as e:
        print(f"Error in create_chat_api: {e}")
        return {"error": str(e)}

@app.delete("/api/delete_chat/{conv_id}")
async def delete_chat_api(conv_id: int):
    """API endpoint for deleting a chat session"""
    try:
        if conv_id in chat_sessions:
            del chat_sessions[conv_id]
            # Force garbage collection after deleting chat
            gc.collect()
            return {"message": f"Chat session {conv_id} deleted successfully"}
        else:
            return {"error": f"Chat session {conv_id} not found"}
    except Exception as e:
        print(f"Error deleting chat {conv_id}: {e}")
        return {"error": str(e)}
    
@app.post("/api/define_chat_name")
async def define_chat_name_api(request: ChatRequest):
    """API Endpoint for defining a chat name based on conversation ID"""
    try:
        # Call define chat name function
        result = define_chat_name(request.conv_id)
        
        return {"result": result}
    except Exception as e:
        print(f"Error in define_chat_name_api: {e}")
        return {"error": str(e)}
    
@app.post("/api/refresh_knowledge")
async def refresh_knowledge_api(request: RefreshKnowledgeRequest, background_tasks: BackgroundTasks):
    """API endpoint for refreshing knowledge base from Laravel - responds immediately then processes in background"""
    try:
        # Check if refresh is already running
        if refresh_status["is_running"]:
            return {
                "message": "Knowledge base refresh is already in progress",
                "status": "already_running",
                "start_time": refresh_status["start_time"].isoformat() if refresh_status["start_time"] else None
            }
        
        # add background task
        background_tasks.add_task(refresh_knowledge_background)
        
        # Respond immediately
        return {
            "message": "Knowledge base refresh started successfully",
            "status": "processing",
            "note": "The refresh process is running in the background"
        }
    except Exception as e:
        print(f"Error in refresh_knowledge_api: {e}")
        return {"error": str(e)}

@app.get("/api/refresh_status")
async def get_refresh_status():
    """API endpoint for checking refresh status"""
    return {
        "is_running": refresh_status["is_running"],
        "last_refresh": refresh_status["last_refresh"],
        "files_processed": refresh_status["files_processed"],
        "error": refresh_status["error"],
        "start_time": refresh_status["start_time"].isoformat() if refresh_status["start_time"] else None,
        "end_time": refresh_status["end_time"].isoformat() if refresh_status["end_time"] else None
    }

@app.post("/api/process_prompt")
async def process_prompt_api(request: PromptRequest):
    """API endpoint for processing text prompts"""
    try:
        result = process_prompt(request.prompt, request.conv_id, request.history)
        return {"result": result}
    except Exception as e:
        print(f"Error in process_prompt_api: {e}")
        return {"error": str(e)}

@app.post("/api/process_files_and_prompt")
async def process_files_and_prompt_api(
    files: List[UploadFile] = File(...),
    custom_prompt: str = Form(...),
    conv_id: int = Form(...),
    history: Optional[str] = Form(None)
):
    """API endpoint for processing files and prompts"""
    temp_dir = None
    temp_files_for_processing = []

    try:
        # Create flag to check if transcript file is present
        should_process_transcript = False
        for file in files:
            # Check filename contains 'transcript'
            if file.filename and 'transcript' in file.filename.lower():
                should_process_transcript = True
                print(f"Detected transcript file: {file.filename}")
                break
        
        def sanitize_filename(filename):
            """Sanitize filename to avoid filesystem issues"""
            if not filename:
                return 'unnamed_file'
            
            # Normalize unicode characters
            filename = unicodedata.normalize('NFKD', filename)
            # Remove or replace problematic characters
            filename = re.sub(r'[^\w\s.-]', '_', filename)
            # Replace multiple spaces/underscores with single underscore
            filename = re.sub(r'[\s_]+', '_', filename)
            # Remove leading/trailing dots and spaces
            filename = filename.strip('. ')
            return filename if filename else 'unnamed_file'

        temp_dir = tempfile.mkdtemp()
        
        for i, file in enumerate(files):
            # Generate a unique temporary filename
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ''
            safe_filename = sanitize_filename(os.path.splitext(file.filename)[0] if file.filename else 'file')
            temp_filename = f"{uuid.uuid4().hex}_{safe_filename}{file_extension}"
            temp_path = os.path.join(temp_dir, temp_filename)

            try:
                content = await file.read()
                                
                if not content:
                    print(f"Warning: File {file.filename} is empty")
                    continue
                
                with open(temp_path, "wb") as buffer:
                    buffer.write(content)
                
                # Verify file was written successfully
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    # Create a simple file object
                    class SimpleFile:
                        def __init__(self, path, original_filename):
                            self.name = path
                            self.filename = original_filename or 'unnamed_file'
                        
                    temp_files_for_processing.append(SimpleFile(temp_path, file.filename))
                else:
                    print(f"Error: Failed to save file {file.filename} to {temp_path}")
                    
            except Exception as file_error:
                print(f"Error saving file {file.filename}: {str(file_error)}")
                continue
        
        if not temp_files_for_processing:
            return {"error": "No valid files were uploaded or saved"}
        
        # Parse history string if provided
        parsed_history = None
        if history:
            try:
                # Convert JSON string back to Python list of dicts
                parsed_history = json.loads(history) # Now parsed_history will be a list of dicts like [{'role': 'user', 'content': '...'}]
            except json.JSONDecodeError:
                print(f"Warning: Could not decode history JSON string: {history}")
                return {"error": "Invalid history format (not a valid JSON string)"}
        
        transcript_config = None
        if should_process_transcript:
            transcript_config = types.GenerateContentConfig(system_instruction=TRANSCRIPT_INSTRUCTION)
            print("Config set to TRANSCRIPT_INSTRUCTION")
        
        # Process the files
        result = process_files_and_prompt(temp_files_for_processing, custom_prompt, conv_id, transcript_config, parsed_history) 
        return {"result": result}
        
    except Exception as e:
        print(f"Error in process_files_and_prompt_api: {e}")
        return {"error": str(e)}
    finally:
        # Cleanup
        for temp_file in temp_files_for_processing: 
            try:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
            except Exception as cleanup_error:
                print(f"Error cleaning up {temp_file.name}: {str(cleanup_error)}")

        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                print(f"Error removing temp directory {temp_dir}: {str(cleanup_error)}")
        
        # Force garbage collection after file processing
        gc.collect()

# Launch the app
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=300,
        limit_concurrency=10,
        access_log=True
    )