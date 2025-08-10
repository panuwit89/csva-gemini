import os
import base64
import tempfile
from google import genai
from google.genai import types

from . import global_state
from .config import LARAVEL_BASE_URL, SYSTEM_INSTRUCTION, GEMINI_API_KEY
from .schemas import Message

client = genai.Client(api_key=GEMINI_API_KEY)

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
                temperature=0.2,
            ),
            history=history,
        )
        global_state.chat_sessions[conv_id] = chat
        print(f"Chat session {conv_id} created successfully.")
        return chat
    except Exception as e:
        print(f"Error creating chat session {conv_id}: {e}")
        raise e

def get_chat_session(conv_id: int, history: list | None = None):
    """
    Gets or recreates a chat session, correctly handling multi-part messages
    from the same role, file upload workarounds, and ensuring history starts correctly.
    """
    if conv_id in global_state.chat_sessions:
        print(f"Found chat session {conv_id} in memory.")
        return global_state.chat_sessions[conv_id]
    
    print(f"Chat session {conv_id} not in memory. Recreating session...")
        
    try:
        recreated_history = []
        if history:
            for msg in history:
                # 1. Process the current message to get its role and parts
                current_parts = []
                # Extract data from either dict or pydantic model
                if isinstance(msg, dict):
                    role_val = msg.get('role', 'user')
                    content_val = msg.get('content', '')
                    attachments_list = msg.get('attachments', [])
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role_val = msg.role
                    content_val = msg.content
                    attachments_list = msg.attachments
                else:
                    print(f"WARNING: Skipping malformed history item: {msg}")
                    continue

                if content_val:
                    current_parts.append(types.Part(text=content_val))
                
                if attachments_list:
                    for attachment_info in attachments_list:
                        # This logic now correctly handles dicts and objects
                        if isinstance(attachment_info, dict):
                            original_name, mime_type, content_data = (
                                attachment_info.get('original_name'),
                                attachment_info.get('mime_type'),
                                attachment_info.get('content_base64')
                            )
                        else:
                            original_name, mime_type, content_data = (
                                attachment_info.original_name,
                                attachment_info.mime_type,
                                attachment_info.content_base64
                            )
                        
                        if not content_data: continue

                        file_content_bytes = base64.b64decode(content_data) if isinstance(content_data, str) else content_data
                        
                        temp_file_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{original_name}") as temp_f:
                                temp_f.write(file_content_bytes)
                                temp_file_path = temp_f.name
                            
                            gemini_file = client.files.upload(file=temp_file_path, config=dict(mime_type=mime_type, display_name=original_name))
                            
                            file_part = types.Part(file_data=types.FileData(mime_type=gemini_file.mime_type, file_uri=gemini_file.uri))
                            current_parts.append(file_part)
                        finally:
                            if temp_file_path and os.path.exists(temp_file_path):
                                os.remove(temp_file_path)

                if not current_parts:
                    continue

                # 2. Merge with previous message if roles are the same
                if recreated_history and recreated_history[-1].role == role_val:
                    recreated_history[-1].parts.extend(current_parts)
                else:
                    recreated_history.append(types.Content(role=role_val, parts=current_parts))

        print(f"Successfully processed history into {len(recreated_history)} turns.")
        
        # 3. FINAL SAFEGUARD: Ensure the history slice passed to the model starts with a user turn.
        start_index = -1
        for i, content in enumerate(recreated_history):
            if content.role == 'user':
                start_index = i
                break
        
        valid_history_for_init = recreated_history[start_index:] if start_index != -1 else None
        
        chat = create_chat_session(conv_id, history=valid_history_for_init)
        
        try:
            initialize_chat_with_docs(chat)
            print(f"Chat session {conv_id} initialized with knowledge documents")
        except Exception as doc_error:
            print(f"Warning: Could not initialize chat {conv_id} with documents: {doc_error}")
        
        return chat

    except Exception as e:
        print(f"An unexpected error occurred while recreating session {conv_id}: {e}")
        raise e

def initialize_chat_with_docs(chat):
    """Initialize a chat session with documents"""
    try:
        if global_state.knowledge_contents:
            chat.send_message(global_state.knowledge_contents)
            print("Chat initialized with knowledge documents")
        else:
            print("No knowledge contents available for initialization")
    except Exception as e:
        print(f"Warning: Could not initialize chat with documents: {e}")

def define_chat_name(conv_id: int):
    """Define a name for the chat session based on the first actual user interaction"""
    try:
        if conv_id not in global_state.chat_sessions:
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