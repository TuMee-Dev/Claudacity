import logging
import os
import shutil
import tempfile
import time
import uuid
from pydantic import BaseModel # type: ignore
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Map of conversation IDs to their temporary directories
conversation_temp_dirs = {}

# Directory for storing temporary conversation directories
CONV_TEMP_ROOT = os.path.join(tempfile.gettempdir(), "claude_conversations")
os.makedirs(CONV_TEMP_ROOT, exist_ok=True)

DEFAULT_MODEL = "anthropic/claude-3.7-sonnet"  # Default Claude model to report (must match /models endpoint)

# Pydantic models for request/response validation

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    format: Optional[str] = None
    template: Optional[str] = None
    keep_alive: Optional[str] = None
    id: Optional[str] = None  # Request ID
    conversation_id: Optional[str] = None  # Explicit conversation ID
    tools: Optional[List[Dict[str, Any]]] = None  # Add tools field for function/tool calling

# OpenAI-compatible models for request validation

class OpenAIChatMessage_old(BaseModel):
    role: str
    content: str

class OpenAIChatRequest_old(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[OpenAIChatMessage_old]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    id: Optional[str] = None  # Request ID
    user: Optional[str] = None  # OpenAI field
    conversation_id: Optional[str] = None  # Explicit conversation ID
    ollama_client: Optional[bool] = False  # Flag for Ollama client


# Helper function to handle Ollama model name convention
def get_ollama_model_name(model_name: str) -> str:
    """
    Apply Ollama model naming conventions:
    - Extract base model name for internal processing by stripping tags (everything after ":")
    - Add ":latest" tag if no tag is present in the model name
    
    Args:
        model_name: Original model name, may include a tag (e.g. "model:tag")
        
    Returns:
        Ollama-formatted model name with appropriate tag
    """
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name

def get_conversation_temp_dir(conversation_id: str) -> str:
    """Get or create a temporary directory for a conversation."""
    if conversation_id in conversation_temp_dirs:
        # Return existing temp dir
        temp_dir = conversation_temp_dirs[conversation_id]
        if os.path.exists(temp_dir):
            return temp_dir
    
    # Create a new temp dir for this conversation
    temp_dir = os.path.join(CONV_TEMP_ROOT, f"conv_{conversation_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)
    conversation_temp_dirs[conversation_id] = temp_dir
    logger.info(f"Created temporary directory for conversation {conversation_id}: {temp_dir}")
    return temp_dir

def cleanup_conversation_temp_dir(conversation_id: str):
    """Clean up the temporary directory for a conversation."""
    if conversation_id in conversation_temp_dirs:
        temp_dir = conversation_temp_dirs[conversation_id]
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory for conversation {conversation_id}: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {temp_dir}: {e}")
        # Remove from the map regardless of success
        del conversation_temp_dirs[conversation_id]

def format_messages_for_claude(request: ChatRequest) -> str:
    """Format messages from Ollama request into a prompt for Claude Code CLI."""
    prompt = ""

    # Add system message if present
    if request.system:
        prompt += f"System: {request.system}\n\n"

    # Add all messages in conversation
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"Human: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
        elif msg.role == "system" and not request.system:
            prompt += f"System: {msg.content}\n\n"

    return prompt

def create_auth_error_response(error_message):
    """Create a standardized Claude authentication error response"""
    return {
        "error": {
            "message": error_message,
            "type": "auth_error",
            "param": None,
            "code": "claude_auth_required"
        },
        "auth_required": True,
        "user_message": "Authentication required: Please log in using the Claude CLI."
    }



