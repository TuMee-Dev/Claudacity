# Claude Code Context for Claudacity Project

## Project Overview
Claudacity is a FastAPI-based server that provides an Ollama-compatible API for Claude Code with cross-platform service management capabilities. It translates requests between the Ollama format and Claude Code's CLI format.

## Key Components
- **claude_ollama_server.py**: Main server implementation
- **OPENWEBUI_COMPATIBILITY.md**: Documentation for OpenWebUI compatibility features
- **tests/**: Contains test files including test_api.py, test_metrics.py, test_dashboard.py

## Important Testing Instructions
- **ALWAYS RUN TESTS AFTER MAKING CHANGES** before telling the user that changes are ready
- Run tests with: `python tests/run_tests.py --all` or for specific tests:
  - API tests: `python tests/run_tests.py --api`
  - Metrics tests: `python tests/run_tests.py --unit`
  - Dashboard tests: `python tests/run_tests.py --dashboard`

## OpenWebUI Compatibility
The project includes special handling for OpenWebUI compatibility, particularly:
1. **Tool/Function Calling Format Conversion**: Converts Claude's `tools` format to OpenWebUI's expected `tool_calls` format
2. **Claude CLI Command Format**: Ensures conversation flag `-c` appears before prompt flag `-p`
3. **Conversation Isolation**: Each conversation has its own temporary directory
4. **Streaming Response Formatting**: Proper SSE formatting with completion signals

## Debugging Notes
- Verify tool format conversion in `format_to_openai_chat_completion` function
- Check temporary directory handling in `get_conversation_temp_dir` and related functions
- For command format issues, examine `run_claude_command` and `stream_claude_output` functions

## Testing Tool/Function Calls
The test_api.py file contains specialized tests for tool/function calling:
- `test_tool_calls_format_conversion`: Tests basic tool conversion
- `test_complex_tool_calls_format_conversion`: Tests multiple tools and complex arguments
- `test_edge_case_tool_calls_format_conversion`: Tests edge cases like malformed JSON

## Future Work Recommendations
- Add comprehensive streaming response tests
- Implement more robust error handling for tool/function calls
- Enhance documentation with detailed API references
- Consider performance optimizations for high-volume usage