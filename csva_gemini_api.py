from typing import List, Dict
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, Form
import re
import os
import uuid
import tempfile
import unicodedata
import traceback
import pathlib
import uvicorn
import requests
import json

# System instruction
SYSTEM_INSTRUCTION = """วัตถุประสงค์และเป้าหมาย: 
* ตอบคำถามของนิสิตภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเกษตรศาสตร์ เกี่ยวกับเรื่องต่างๆ ที่เกี่ยวข้องกับการศึกษาในภาควิชาและมหาวิทยาลัย.
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
* หากข้อมูลที่นิสิตสอบถามไม่อยู่ในไฟล์ความรู้ ให้ค้นหาข้อมูลที่มีความใกล้เคียงหรือเกี่ยวข้องที่สุด จากแหล่งข้อมูลอื่น.
* นอกเหนือจากนี้ ให้แจ้งว่าไม่สามารถให้ข้อมูลในส่วนนั้นได้โดยตรง แต่สามารถให้ข้อมูลอื่นที่เกี่ยวข้องได้."""

# Base URL for Laravel API
LARAVEL_BASE_URL = "http://localhost"

class PromptRequest(BaseModel):
    prompt: str
    conv_id: int

class ChatRequest(BaseModel):
    conv_id: int
    
class RefreshKnowledgeRequest(BaseModel):
    force: bool = False

# Initialize the Gemini client with provided API key
client = genai.Client(api_key="AIzaSyB67xn_olcqoGAn-IAvdFTTeuhGaaEBiEY")

# Dictionary to store chat sessions
chat_sessions: Dict[int, object] = {}

# Global variable to store processed knowledge files
knowledge_contents = []

# Alternative method using pathlib
base_dir = pathlib.Path.cwd() / "doc"
pdf_files = list(base_dir.glob("*.pdf"))

def fetch_active_knowledge_files():
    """Fetch active knowledge files from Laravel API"""
    try:
        response = requests.get(f"{LARAVEL_BASE_URL}/api/knowledge/active")
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
        # แก้ไข URL ให้ตรงกับ Laravel storage URL
        file_url = f"{LARAVEL_BASE_URL}/storage/{file_path}"
        
        response = requests.get(file_url)
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
    uploaded_files = []
    
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
            
            # Create Gemini Part from file content
            part = types.Part.from_bytes(
                data=file_content,
                mime_type='application/pdf',
            )
            contents.append(part)
            
            # Upload file to Gemini for processing
            # Create temporary file for upload
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, filename)
            
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file_content)
            
            # Upload to Gemini
            uploaded_file = client.files.upload(
                file=temp_file_path,
                config=dict(mime_type='application/pdf')
            )
            uploaded_files.append(uploaded_file)
            
            # Clean up temp file
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
            
            print(f"Processed: {title} ({filename})")
            
        except Exception as e:
            print(f"Error processing knowledge file {knowledge.get('title', 'Unknown')}: {e}")
            continue
    
    print(f"Successfully processed {len(contents)} knowledge files")
    return contents

def refresh_knowledge_base():
    """Refresh knowledge base from Laravel"""
    global knowledge_contents
    
    print("Refreshing knowledge base from Laravel...")
    knowledge_contents = process_knowledge_files_from_laravel()
    
    # Update all existing chat sessions with new knowledge
    for conv_id, chat in chat_sessions.items():
        try:
            if knowledge_contents:
                chat.send_message(knowledge_contents)
                print(f"Updated chat session {conv_id} with new knowledge")
        except Exception as e:
            print(f"Error updating chat session {conv_id}: {e}")
    
    return len(knowledge_contents)

def create_chat_session(conv_id: int):
    """Create a new chat session with the Gemini model and configuration"""
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        )
    )
    chat_sessions[conv_id] = chat
    return chat

def get_chat_session(conv_id: int):
    """Get existing chat by conversation ID"""
    if conv_id not in chat_sessions:
        raise ValueError(f"Chat session '{conv_id}' not found")
    
    return chat_sessions[conv_id]

def process_all_files(pdf_files):
    """Process all PDF files in a single request"""
    if not pdf_files:
        print("No files to process")
        return
    
    # Create content list with all files
    contents = []
    
    # Add all files to contents
    for pdf_file in pdf_files:
        filepath = pathlib.Path(pdf_file)
        try:
            contents.append(
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                )
            )
            print(f"Added: {filepath.name}")
            
            # Upload file to Gemini by passing the file path directly
            client.files.upload(
                file=filepath,
                config=dict(mime_type='application/pdf')
            )
            
        except Exception as e:
            print(f"Error adding {filepath.name}: {e}")
    
    return contents

def process_prompt(prompt, conv_id):
    """
    Process a text prompt
    """
    try:
        # Get the chat session for the conversation ID
        chat = get_chat_session(conv_id)

        # Send the prompt to Gemini and get the response
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def process_files_and_prompt(files, custom_prompt, conv_id):
    """
    Process uploaded files and a prompt
    """
    try:
        # Get the chat session for the conversation ID
        chat = get_chat_session(conv_id)

        # List to store uploaded files (Gemini File objects)
        uploaded_gemini_files = []

        # Process each uploaded file
        for file in files:
            # Determine file type based on extension
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == '.pdf':
                mime_type = 'application/pdf'
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                mime_type = f'image/{file_extension[1:]}'
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
        
        # If no files were uploaded, return an error
        if not uploaded_gemini_files:
            return "No files were uploaded. Please upload at least one file."
        
        chat.send_message(uploaded_gemini_files)
        response = chat.send_message(custom_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def initialize_chat_with_docs(chat):
    """Initialize a chat session with documents"""
    if knowledge_contents:
        chat.send_message(knowledge_contents)

# Initialize knowledge base on startup
print("Initializing knowledge base from Laravel...")
knowledge_contents = process_knowledge_files_from_laravel()
        
# # Process all PDF files and send the contents to Gemini
# my_contents = process_all_files(pdf_files)

# Create FastAPI app for API endpoints
app = FastAPI()

@app.post("/api/create_chat")
async def create_chat_api(request: ChatRequest):
    """API endpoint for creating a new chat session"""
    try:
        if request.conv_id in chat_sessions:
            return {"error": f"Chat session {request.conv_id} already exists"}
        
        chat = create_chat_session(request.conv_id)
        initialize_chat_with_docs(chat)
        return {"message": f"Chat session {request.conv_id} created successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/api/refresh_knowledge")
async def refresh_knowledge_api(request: RefreshKnowledgeRequest):
    """API endpoint for refreshing knowledge base from Laravel"""
    try:
        count = refresh_knowledge_base()
        return {
            "message": f"Knowledge base refreshed successfully",
            "files_processed": count
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/knowledge_status")
async def knowledge_status_api():
    """API endpoint for checking knowledge base status"""
    return {
        "knowledge_files_loaded": len(knowledge_contents),
        "active_chat_sessions": len(chat_sessions)
    }    
    
@app.get("/api/list_chats")
async def list_chats_api():
    """API endpoint for listing all active chat sessions"""
    return {"chat_sessions": list(chat_sessions.keys())}

@app.delete("/api/delete_chat/{conv_id}")
async def delete_chat_api(conv_id: int):
    """API endpoint for deleting a chat session"""
    if conv_id in chat_sessions:
        del chat_sessions[conv_id]
        return {"message": f"Chat session {conv_id} deleted successfully"}
    else:
        return {"error": f"Chat session {conv_id} not found"}

@app.post("/api/process_prompt")
async def process_prompt_api(request: PromptRequest):
    """API endpoint for processing text prompts"""
    try:
        result = process_prompt(request.prompt, request.conv_id)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/process_files_and_prompt")
async def process_files_and_prompt_api(
    files: List[UploadFile] = File(...),
    custom_prompt: str = Form(...),
    conv_id: int = Form(...)
):
    """API endpoint for processing files and prompts"""
    temp_dir = None # Initialize temp_dir outside try for proper cleanup

    try:
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

        # Save uploaded files temporarily with safe filenames
        temp_files_for_processing = [] # List to store SimpleFile objects for your process_files_and_prompt
        temp_dir = tempfile.mkdtemp()
        
        for i, file in enumerate(files):
            # Generate a unique temporary filename
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ''
            safe_filename = sanitize_filename(os.path.splitext(file.filename)[0] if file.filename else 'file')
            temp_filename = f"{uuid.uuid4().hex}_{safe_filename}{file_extension}"
            temp_path = os.path.join(temp_dir, temp_filename)

            # Save the file
            try:
                content = await file.read()
                                
                if not content:
                    print(f"Warning: File {file.filename} is empty")
                    continue
                
                with open(temp_path, "wb") as buffer:
                    buffer.write(content)
                
                # Verify file was written successfully
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    file_size = os.path.getsize(temp_path)
                    
                    # Create a simple file object that matches what your process function expects
                    class SimpleFile:
                        def __init__(self, path, original_filename):
                            self.name = path
                            self.filename = original_filename or 'unnamed_file'
                        
                    temp_files_for_processing.append(SimpleFile(temp_path, file.filename))
                else:
                    print(f"Error: Failed to save file {file.filename} to {temp_path}")
                    
            except Exception as file_error:
                print(f"Error saving file {file.filename}: {str(file_error)}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not temp_files_for_processing:
            error_msg = "No valid files were uploaded or saved"
            return {"error": error_msg}
        
        # List all temp files before processing
        for temp_file in temp_files_for_processing:
            print(f"Temp file: {temp_file.filename} -> {temp_file.name}")
        
        # Process the files
        try:
            # Pass the list of SimpleFile objects
            result = process_files_and_prompt(temp_files_for_processing, custom_prompt, conv_id) 
            if isinstance(result, str) and len(result) > 100:
                print(f"Result preview: {result[:100]}...")
            else:
                print(f"Result: {result}")
                
        except Exception as process_error:
            error_msg = f"Error processing files: {str(process_error)}"
            return {"error": error_msg}
        
        return {"result": result}
        
    except Exception as e:
        error_msg = f"Unexpected error in process_files_and_prompt_api: {str(e)}"
        return {"error": error_msg}
    finally:
        # Cleanup temporary files and directory
        for temp_file in temp_files_for_processing: 
            try:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
            except Exception as cleanup_error:
                print(f"Error cleaning up {temp_file.name}: {str(cleanup_error)}")

        if temp_dir and os.path.exists(temp_dir): # Check if temp_dir exists before trying to remove
            try:
                os.rmdir(temp_dir)
                print(f"Removed temp directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Error removing temp directory {temp_dir}: {str(cleanup_error)}")

# Launch the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)