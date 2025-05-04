# Claude Ollama API Server

A FastAPI-based server that provides an Ollama-compatible API for Claude Code.

## Overview

This server allows you to use Claude Code through the Ollama API interface. It translates requests between the Ollama format and Claude Code's CLI format, enabling:

- Chat completions with Claude 3.7 models
- 128k context window support
- True streaming responses (using Claude's streaming JSON output)
- Conversation tracking and memory
- Ollama-compatible API endpoints

## Prerequisites

- Claude Code must be installed and configured on your system
- The `claude` command must be available in your PATH

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python claude_ollama_server.py
```

By default, the server runs on `http://0.0.0.0:22434` (listening on all network interfaces) to avoid conflicts with Ollama's default port.

You can specify a different port as a command-line argument:

```bash
python claude_ollama_server.py 22435
```

## API Endpoints

### Primary API Endpoints (OpenAI-compatible)
- `/chat/completions` - Main chat completions endpoint 
- `/models` - List available models
- `/version` - Get server version information

### OpenAI-compatible API (for OpenAI clients)
- `/v1/chat/completions` - OpenAI-compatible chat completions
- `/v1/models` - List available models in OpenAI format

### Other
- `/docs` - Interactive API documentation (provided by FastAPI)

## Example Requests

### Primary API
```bash
curl http://localhost:22434/chat/completions -d '{
  "model": "claude-3-7-sonnet-latest",
  "messages": [
    { "role": "user", "content": "Write a haiku about programming" }
  ],
  "id": "conversation-123"
}'
```

The `id` field is optional but recommended for maintaining conversation context.

### OpenAI-compatible API
```bash
curl http://localhost:22434/v1/chat/completions -d '{
  "model": "claude-3-7-sonnet-latest",
  "messages": [
    { "role": "user", "content": "Write a haiku about programming" }
  ],
  "id": "conversation-123"
}'
```

The `id` field is the primary field for conversation tracking. The `user` field can also be used as a fallback if `id` is not provided.

## Environment Variables

- `HOST` - Host to bind the server to (default: 127.0.0.1)
- `PORT` - Port to run the server on (default: 22434)

## Client Compatibility

### Ollama Client
```python
from ollama import Client

client = Client(host="http://localhost:22434")
response = client.chat(
    model="claude-3-7-sonnet-latest",
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)
```

### OpenAI Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:22434/v1/",
    api_key="not-needed-but-required-by-client"
)

response = client.chat.completions.create(
    model="claude-3-7-sonnet-latest",
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)
```

### Open WebUI
To connect Open WebUI to this server, set:
- OLLAMA_BASE_URL=http://localhost:22434

## How It Works

The server intercepts Ollama API requests and:

1. Converts the request format to Claude Code's expected format
2. Launches Claude Code with the appropriate flags:
   - For streaming: `claude -p --output-format stream-json`
   - For non-streaming: `claude -p --output-format json`
   - For conversations: adds `-c <conversation_id>` to maintain context
3. Passes the user's prompt to Claude Code
4. For streaming requests, captures Claude's streaming JSON output in real-time
5. Formats the responses in Ollama's expected format
6. Returns responses to the client with proper SSE formatting

## Limitations

- Some Ollama-specific features may not be fully supported
- Performance metrics are simulated as Claude doesn't provide the same metrics as Ollama
- The server requires Claude Code to be installed and configured

## License

MIT