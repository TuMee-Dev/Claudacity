# Ollama API Compatibility

This document outlines the Ollama API compatibility layer implemented in the Claude proxy server. This compatibility layer allows tools and applications designed for Ollama to work with Claude without modification.

## Supported Endpoints

The following Ollama-compatible endpoints are implemented:

### `/api/version`

Returns version information in the Ollama format.

**Request:**
```http
GET /api/version
```

**Response:**
```json
{
  "version": "0.1.0"
}
```

### `/api/tags`

Returns a list of available models in the Ollama format.

**Request:**
```http
GET /api/tags
```

**Response:**
```json
{
  "models": [
    {
      "name": "claude-3.7-sonnet",
      "model": "claude-3.7-sonnet",
      "modified_at": "2025-05-05T19:53:25.564072",
      "size": 0,
      "digest": "anthropic_claude_3_7_sonnet_20250505",
      "details": {
        "parent_model": "",
        "format": "api",
        "model": "claude-3.7-sonnet",
        "family": "anthropic",
        "families": ["anthropic", "claude"],
        "parameter_size": "13B",
        "quantization_level": "none"
      }
    }
  ]
}
```

### `/api/chat`

Handles chat interactions in Ollama format, with streaming support.

**Request:**
```http
POST /api/chat
```

**Request Body:**
```json
{
  "model": "claude-3.7-sonnet",
  "messages": [
    {
      "role": "user",
      "content": "Hello, who are you?"
    }
  ],
  "stream": true,
  "options": {
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

**Streaming Response:**
A series of server-sent events with each chunk in the following format:

```json
{
  "model": "claude-3.7-sonnet",
  "created_at": "2025-05-05T19:55:32.123456Z",
  "message": {
    "role": "assistant",
    "content": "chunk of text"
  },
  "done": false
}
```

Final message in stream:

```json
{
  "model": "claude-3.7-sonnet",
  "created_at": "2025-05-05T19:55:35.123456Z",
  "message": {
    "role": "assistant",
    "content": ""
  },
  "done_reason": "stop",
  "done": true,
  "total_duration": 3123456,
  "load_duration": 312345,
  "prompt_eval_count": 123,
  "prompt_eval_duration": 1123456,
  "eval_count": 456,
  "eval_duration": 1687655
}
```

**Non-streaming Response:**
```json
{
  "model": "claude-3.7-sonnet",
  "created_at": "2025-05-05T19:55:32.123456Z",
  "message": {
    "role": "assistant",
    "content": "Hello! I'm Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. I'm designed to assist with a wide variety of tasks including answering questions, generating creative content, helping with writing and research, and much more. How can I help you today?"
  },
  "done_reason": "stop",
  "done": true,
  "total_duration": 3123456,
  "load_duration": 312345,
  "prompt_eval_count": 123,
  "prompt_eval_duration": 1123456,
  "eval_count": 456,
  "eval_duration": 1687655
}
```

### `/api/generate`

Provides completion functionality in Ollama format.

**Request:**
```http
POST /api/generate
```

**Request Body:**
```json
{
  "model": "claude-3.7-sonnet",
  "prompt": "Once upon a time,",
  "stream": true,
  "options": {
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

**Response:**
Same format as `/api/chat` responses.

## Implementation Details

- The Ollama compatibility layer reuses the existing Claude integration.
- Response formats match Ollama's expected format as closely as possible.
- The `/api/generate` endpoint converts prompts to chat messages and reuses the chat implementation.
- Both streaming and non-streaming responses are supported.
- Claude models are presented in Ollama-compatible format.

## Usage with Ollama Clients

Any client application designed to work with Ollama should be able to connect to this API by simply changing the base URL to point to this server instead of an Ollama installation.

Example with `curl`:

```bash
# Using Ollama directly
curl -X POST http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [{"role": "user", "content": "Hello, who are you?"}]
}'

# Using Claude with Ollama compatibility
curl -X POST http://localhost:8000/api/chat -d '{
  "model": "claude-3.7-sonnet",
  "messages": [{"role": "user", "content": "Hello, who are you?"}]
}'
```

## Limitations

- Only Claude models are available through this API.
- Model-specific parameters like context length depend on the underlying Claude model.
- Some Ollama-specific features (like model downloading) are not applicable.