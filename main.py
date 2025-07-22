import uvicorn
from app.routes import app # Import app instance จาก routes.py
from app.knowledge_management import refresh_knowledge_base

# Initialize knowledge base on startup
print("Initializing knowledge base from Laravel...")
try:
    refresh_knowledge_base()
    print(f"Startup knowledge loading complete.")
except Exception as e:
    print(f"Error during startup knowledge loading: {e}")

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