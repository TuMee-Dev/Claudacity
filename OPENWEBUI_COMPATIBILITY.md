# OpenWebUI Compatibility Guide

This document explains fixes made to ensure compatibility between the Claude Ollama API server and OpenWebUI.

## CRITICAL: Conversation ID Management

**IMPORTANT**: The server should never generate its own conversation IDs except when explicitly needed for multipart conversations.

- Only use conversation IDs provided by OpenWebUI in the request
- Never generate fallback random conversation IDs as they can leak into user prompts
- Only pass conversation IDs to Claude CLI when necessary (multipart conversations or tools)
- For single message exchanges, it's better to omit the conversation ID than risk leakage
- Leaking conversation IDs into prompts causes Claude to respond with "I notice you've mentioned a conversation ID"

## Key Compatibility Issues Fixed

0. **Streaming Response Completion**: Fixed OpenWebUI not receiving proper completion signals for streaming
   - Enhanced SSE (Server-Sent Events) formatting with proper delimiters
   - Added small delays between final chunk and DONE marker to ensure proper order
   - Improved error handling to always send the DONE marker
   - Added extra logging to track completion signals
   - Applied robust formatting to all streaming responses

1. **Tool/Function Calling Flow**: Fixed OpenWebUI function calling flow for proper tools response handling
   - Detects JSON responses containing Claude's tools array and converts them to OpenWebUI's expected tool_calls format
   - Transforms Claude's {"tools": [{...}]} format to OpenWebUI's {"tool_calls": [{...}]} format
   - Properly extracts tool names and parameters from Claude's function calling syntax
   - This transformation is necessary because OpenWebUI expects a specific tool_calls format
   - Added extensive logging to track function calling responses
   - Applied to both streaming and non-streaming responses

2. **Model ID Consistency**: Ensured consistent model ID format between /models endpoint and request handling
   - Model ID standardized to "anthropic/claude-3.7-sonnet" across the codebase
   - Previously, there was an inconsistency where some parts used "claude-3-7-sonnet-latest"

3. **Streaming Response Format**: Improved Server-Sent Events (SSE) format for streaming responses
   - Now using standard FastAPI StreamingResponse with appropriate headers
   - Added required SSE format with correct "data:" prefix
   - Ensured proper [DONE] marker is sent at the end of each response
   - Added double newlines after each SSE chunk to follow the SSE specification

4. **CORS and Streaming Headers**: Added necessary headers for streaming responses
   - Added Cache-Control: no-cache
   - Added Connection: keep-alive
   - Added Access-Control-Allow-Origin: *
   - Added X-Accel-Buffering: no (helps prevent buffering in Nginx proxies)

5. **Enhanced Logging**: Added detailed logging for easier debugging of OpenWebUI requests
   - Logs User-Agent and Origin for each chat completion request
   - Logs model and stream preference for each request
   - Logs Content-Type for each response
   - Records full request/response cycle information
   - Added special logging for streaming completion signals to track issues

## Testing

The fixes have been tested with:

1. Non-streaming requests using curl:
   ```bash
   curl -X POST http://localhost:22434/chat/completions -H "Content-Type: application/json" -d '{"model":"anthropic/claude-3.7-sonnet","messages":[{"role":"user","content":"This is a test respond with only OK"}],"stream":false}'
   ```

2. Streaming requests using curl:
   ```bash
   curl -X POST http://localhost:22434/chat/completions -H "Content-Type: application/json" -d '{"model":"anthropic/claude-3.7-sonnet","messages":[{"role":"user","content":"This is a test respond with only OK"}],"stream":true}' -N
   ```

## Troubleshooting

If OpenWebUI still doesn't work with the Claude Ollama API server:

1. Check the logs in `/Users/bsobel/Projects/Claudacity/logs/stderr.log` for any error messages
2. Ensure OpenWebUI is correctly configured to use the API server at http://localhost:22434
3. Verify the model selection in OpenWebUI matches the model ID "anthropic/claude-3.7-sonnet"
4. Try restarting both the Claude API server and OpenWebUI

To restart the Claude API server:

```bash
cd /Users/bsobel/Projects/Claudacity
python claude_service.py --stop
python claude_service.py --start
```