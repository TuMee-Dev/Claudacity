import models
import time
import logging
import config

logger = logging.getLogger(__name__)

# Conversation cache to track active conversations
# Keys are conversation IDs, values are (timestamp, conversation_id) tuples
conversation_cache = {}

def conversation_cache_cleanup():
    """Periodically clean up old data to prevent memory leaks."""
    # Clean up old conversation temp directories that are no longer active
    try:
        active_conversations = set()
        # Get active conversations from cache
        for request_id, (_, conv_id) in conversation_cache.items():
            active_conversations.add(conv_id)
        
        # Clean up temp directories for inactive conversations
        for conv_id in list(models.conversation_temp_dirs.keys()):
            if conv_id not in active_conversations:
                models.cleanup_conversation_temp_dir(conv_id)
                logger.debug(f"Cleaned up temp directory for inactive conversation: {conv_id}")
    except Exception as e:
        logger.error(f"Error cleaning up conversation temp dirs: {e}")

# Conversation management functions

def get_conversation_id(request_id: str) -> str:
    """
    Get the conversation ID for a request. If it exists in the cache,
    return the cached conversation ID. Otherwise, return None.
    Also cleans up expired conversations.
    """
    current_time = time.time()

    # Clean up expired conversations
    expired_keys = []
    for key, (timestamp, _) in conversation_cache.items():
        if current_time - timestamp > config.CONVERSATION_CACHE_TTL:
            expired_keys.append(key)

    for key in expired_keys:
        logger.info(f"Removing expired conversation: {key}")
        del conversation_cache[key]

    # Return the conversation ID if it exists
    if request_id in conversation_cache:
        conversation_cache[request_id] = (current_time, conversation_cache[request_id][1])  # Update timestamp
        return conversation_cache[request_id][1]

    return None

def set_conversation_id(request_id: str, conversation_id: str):
    """
    Store a conversation ID in the cache.
    """
    current_time = time.time()
    conversation_cache[request_id] = (current_time, conversation_id)
    logger.info(f"Stored conversation ID {conversation_id} for request {request_id}")
