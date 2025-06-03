import gradio as gr
from google import genai
from google.genai import types
import os
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import uvicorn
from pydantic import BaseModel
import uuid
import tempfile
import unicodedata
import re
import traceback
import pathlib

class PromptRequest(BaseModel):
    prompt: str

# Initialize the Gemini client with provided API key
client = genai.Client(api_key="")
chat = client.chats.create(model="gemini-2.0-flash")

# Alternative method using pathlib
base_dir = pathlib.Path.cwd() / "doc"
pdf_files = list(base_dir.glob("*.pdf"))

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

def process_prompt(prompt):
    """
    Process a text prompt
    """
    try:
        # Send the prompt to Gemini and get the response
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def process_files_and_prompt(files, custom_prompt):
    """
    Process uploaded files and a prompt
    """
    try:
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
        traceback.print_exc()
        return f"Error: {str(e)}"
    
my_contents = process_all_files(pdf_files)
chat.send_message(my_contents)

# Create the Gradio interface
with gr.Blocks(title="Gemini Visual Assistant") as demo:
    gr.Markdown("# Gemini Visual Assistant")

    with gr.Tab("Process Prompt"):
        gr.Markdown("Enter a text prompt to get a response from Gemini.")
        
        text_prompt = gr.Textbox(label="Your Prompt", placeholder="Ask Gemini a question...", lines=3)
        text_submit = gr.Button("Submit")
        text_output = gr.Textbox(label="Gemini Response", lines=10)
        
        text_submit.click(
            fn=process_prompt,
            inputs=text_prompt,
            outputs=text_output
        )
    
    with gr.Tab("Process Files and Prompt"):
        gr.Markdown("Upload files and ask Gemini questions about them.")
        
        with gr.Row():
            with gr.Column():
                files = gr.File(label="Upload Files", file_count="multiple")
                prompt = gr.Textbox(label="Your Prompt", placeholder="Ask Gemini about these files...", lines=3)
                submit_button = gr.Button("Process Files")
            
            with gr.Column():
                output = gr.Textbox(label="Gemini Response", lines=10)
        
        submit_button.click(
            fn=process_files_and_prompt,
            inputs=[files, prompt],
            outputs=output
        )
        
        gr.Markdown("""
        ## How to Use
        1. Upload one or more files (PDF, images, or text)
        2. Enter your prompt asking about the uploaded files
        3. Click "Process Files" and wait for the response
        
        ## Example Prompts
        - "Summarize the key points from these papers"
        - "Compare the methodologies used in these documents"
        - "Create a table of the main results from these papers"
        """)

# Create FastAPI app for API endpoints
app = FastAPI()

@app.post("/api/process_prompt")
async def process_prompt_api(request: PromptRequest):
    """API endpoint for processing text prompts"""
    try:
        result = process_prompt(request.prompt)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/process_files_and_prompt")
async def process_files_and_prompt_api(
    files: List[UploadFile] = File(...),
    custom_prompt: str = Form(...)
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
            result = process_files_and_prompt(temp_files_for_processing, custom_prompt) 
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

# Mount the Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# Launch the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)