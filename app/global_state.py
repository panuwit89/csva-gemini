import threading
from typing import Dict, List

# Dictionary to store chat sessions
chat_sessions: Dict[int, object] = {}

# Global variable to store processed knowledge files
knowledge_contents: List[object] = []

# Collect refresh status information
refresh_status = {
    "is_running": False,
    "last_refresh": None,
    "files_processed": 0,
    "error": None,
    "start_time": None,
    "end_time": None
}

# Lock for thread safety
refresh_lock = threading.Lock()