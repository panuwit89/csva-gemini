import os
import re
import gc
import json
import uuid
import tempfile
import unicodedata
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from google.genai import types

from . import global_state, gemini_services, knowledge_management
from .schemas import ChatRequest, PromptRequest, RefreshKnowledgeRequest
from .config import TRANSCRIPT_INSTRUCTION

app = FastAPI()

def refresh_knowledge_background():
    """Background task for refreshing knowledge base"""
    try:
        print("Starting background knowledge refresh...")
        count = knowledge_management.refresh_knowledge_base()
        print(f"Background knowledge refresh completed. Files processed: {count}")
    except Exception as e:
        print(f"Error in background knowledge refresh: {e}")

# --- API Endpoints ---
@app.post("/api/create_chat")
async def create_chat_api(request: ChatRequest, background_tasks: BackgroundTasks):
    """API endpoint for creating a new chat session"""
    try:
        if request.conv_id in global_state.chat_sessions:
            return {"error": f"Chat session {request.conv_id} already exists"}
        
        chat = gemini_services.create_chat_session(request.conv_id)
        
        background_tasks.add_task(gemini_services.initialize_chat_with_docs, chat)
        
        return {"message": f"Chat session {request.conv_id} created successfully"}
    except Exception as e:
        print(f"Error in create_chat_api: {e}")
        return {"error": str(e)}

@app.delete("/api/delete_chat/{conv_id}")
async def delete_chat_api(conv_id: int):
    """API endpoint for deleting a chat session"""
    try:
        if conv_id in global_state.chat_sessions:
            del global_state.chat_sessions[conv_id]
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
        result = gemini_services.define_chat_name(request.conv_id)
        
        return {"result": result}
    except Exception as e:
        print(f"Error in define_chat_name_api: {e}")
        return {"error": str(e)}

@app.post("/api/refresh_knowledge")
async def refresh_knowledge_api(request: RefreshKnowledgeRequest, background_tasks: BackgroundTasks):
    """API endpoint for refreshing knowledge base from Laravel - responds immediately then processes in background"""
    try:
        # Check if refresh is already running
        if global_state.refresh_status["is_running"]:
            return {
                "message": "Knowledge base refresh is already in progress",
                "status": "already_running",
                "start_time": global_state.refresh_status["start_time"].isoformat() if global_state.refresh_status["start_time"] else None
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
        "is_running": global_state.refresh_status["is_running"],
        "last_refresh": global_state.refresh_status["last_refresh"],
        "files_processed": global_state.refresh_status["files_processed"],
        "error": global_state.refresh_status["error"],
        "start_time": global_state.refresh_status["start_time"].isoformat() if global_state.refresh_status["start_time"] else None,
        "end_time": global_state.refresh_status["end_time"].isoformat() if global_state.refresh_status["end_time"] else None
    }

@app.post("/api/process_prompt")
async def process_prompt_api(request: PromptRequest):
    """API endpoint for processing text prompts"""
    try:
        result = gemini_services.process_prompt(request.prompt, request.conv_id, request.history)
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
        result = gemini_services.process_files_and_prompt(temp_files_for_processing, custom_prompt, conv_id, transcript_config, parsed_history) 
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