# Timeout Testing for Claudacity

This directory contains scripts to test and verify the timeout behavior of the Claudacity proxy server, particularly for complex prompts like the "distributed job scheduling system" prompt.

## Test Files

1. **test_complex_prompt.py**
   - Tests both direct Ollama API and the proxy server with a complex prompt
   - Measures timing between chunks of the response
   - Provides detailed logs of chunk timing and delays

2. **run_timeout_test.py**
   - Runner script that executes the tests in test_complex_prompt.py
   - Provides a cleaner way to run the tests

3. **verify_timeout_changes.py**
   - Verifies that the timeout detection and process termination mechanisms are working correctly
   - Creates a simulated hanging process using a sleep command
   - Confirms that hung processes are detected, terminated, and untracked after the configured timeout

## Running the Tests

To verify timeout handling:

```bash
# Run the timeout verification test
python tests/verify_timeout_changes.py
```

To test with actual complex prompts (requires running services):

```bash
# Run both direct and proxy tests
python tests/run_timeout_test.py

# Configure custom endpoints using environment variables
OLLAMA_URL="http://localhost:11434/api/chat" PROXY_URL="http://localhost:8000/api/chat" python tests/run_timeout_test.py
```

## Findings

See the detailed findings in [TIMEOUT_INVESTIGATION_REPORT.md](TIMEOUT_INVESTIGATION_REPORT.md).

Key points:
- The timeout mechanism is working correctly
- Current timeout values (10s chunk timeout / 60s max silence) are appropriate
- Hung processes are properly detected, terminated, and untracked

## Next Steps

1. Consider dynamic timeout adjustment based on prompt complexity
2. Improve user feedback when timeouts occur
3. Add more detailed logging and metrics for timeout-related events