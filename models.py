import logging
import os
import shutil
import tempfile
import uuid


logger = logging.getLogger(__name__)

# Map of conversation IDs to their temporary directories
conversation_temp_dirs = {}

# Directory for storing temporary conversation directories
CONV_TEMP_ROOT = os.path.join(tempfile.gettempdir(), "claude_conversations")
os.makedirs(CONV_TEMP_ROOT, exist_ok=True)

DEFAULT_MODEL = "anthropic/claude-3.7-sonnet"  # Default Claude model to report (must match /models endpoint)

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
