DEBUG = False
API_VERSION = "0.1.0"
BUILD_NAME = "claudacity-server"
GIT_SHA = ""
BUILD_DATE = ""
DEFAULT_MAX_TOKENS = 128000 # 128k tokens
CONVERSATION_CACHE_TTL = 3600 * 3
AVAILABLE_MODELS = [
    {
        "name": "anthropic/claude-3.7-sonnet",
        "modified_at": "2025-09-05T19:53:25.564072",
        "size": 0,
        "digest": "anthropic_claude_3_7_sonnet_20250505",
        "details": {
            "model": "anthropic/claude-3.7-sonnet",
            "parent_model": "",
            "format": "api",
            "family": "anthropic",
            "families": ["anthropic", "claude"],
            "parameter_size": "13B",
            "quantization_level": "none"
        }
    }
]