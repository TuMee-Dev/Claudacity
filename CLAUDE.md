# Claude Ollama Development Notes

## Key Development Guidelines
- Always test the code A/B against ollama after changes. Use the test scripts in the tests directory.
- Restart the service before testing it using the service scripts: `python claude_service.py --restart`
- ALWAYS USE THE SERVICE SCRIPTS for stopping, starting and restarting the service.
- Clear out the log files in the logs directory if you need to check the logs.
- Read additional context from the .md files in the context directory.
- Use the unified testing framework to ensure both ollama and claude are working after code fixes
- All code is async, do not user requests library use aiohttp

## Important Files in Context Directory
The context directory contains detailed documentation about various aspects of the system:
- `METRICS_ADAPTER.md` - Information about the metrics adapter system
- `OLLAMA_COMPATIBILITY.md` - Details about Ollama API compatibility
- `OLLAMA_API_IMPLEMENTATION.md` - Detailed implementation notes for the Ollama API
- `OPENWEBUI_COMPATIBILITY.md` - Details about OpenWebUI integration
- `TOOL_FIXES.md` - Documentation about tool handling fixes

For a general README about the Claude Ollama API Server, see `context/claude_ollama_README.md`.

## Port Information
- Ports: 11434 is Ollama, 22434 is our proxy, 334xx is available to you as needed while testing