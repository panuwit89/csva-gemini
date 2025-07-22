import os
import gc
import json
import tempfile
import datetime
import requests
from google import genai

from . import global_state
from .config import LARAVEL_BASE_URL, GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

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

                # Check file extension to determine mime_type
                if filename.lower().endswith('.txt'):
                    mime_type = 'text/plain'  # Mime type for Text files
                elif filename.lower().endswith('.json'):
                    mime_type = 'text/plain' # Mime type for JSON files
                else:
                    mime_type = 'application/pdf' # Mime type for PDF files

                # 2. Upload to Gemini using the determined mime_type
                uploaded_file = client.files.upload(
                    file=temp_file_path,
                    config=dict(mime_type=mime_type)
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
    with global_state.refresh_lock:  # Thread safety
        if global_state.refresh_status["is_running"]:
            print("Refresh already in progress, skipping...")
            return len(global_state.knowledge_contents)

        global_state.refresh_status["is_running"] = True
        global_state.refresh_status["start_time"] = datetime.datetime.now()
        global_state.refresh_status["error"] = None

    try:
        print("Refreshing knowledge base from Laravel...")
        
        # Clear old knowledge contents to free memory
        old_contents = global_state.knowledge_contents
        global_state.knowledge_contents = []
        
        # Force garbage collection
        del old_contents
        gc.collect()
        
        # Process new knowledge files
        new_contents = process_knowledge_files_from_laravel()
        global_state.knowledge_contents = new_contents  # Update global variable
        
        # Edit: not update existing chat sessions
        print("Knowledge base refreshed successfully")
        print("Note: New chat sessions will use updated knowledge. Existing chat session contents remain unchanged.")

        global_state.refresh_status["files_processed"] = len(global_state.knowledge_contents)
        global_state.refresh_status["last_refresh"] = datetime.datetime.now().isoformat()

        return len(global_state.knowledge_contents)

    except Exception as e:
        error_msg = str(e)
        global_state.refresh_status["error"] = error_msg
        print(f"Error refreshing knowledge base: {error_msg}")
        raise e
        
    finally:
        global_state.refresh_status["is_running"] = False
        global_state.refresh_status["end_time"] = datetime.datetime.now()
        # Force garbage collection after refresh
        gc.collect()