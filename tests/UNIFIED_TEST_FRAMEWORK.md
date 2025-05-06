# Claudacity Unified Test Framework

This framework provides a systematic way to test both Ollama and the Claudacity proxy with various prompts, testing both streaming and non-streaming modes.

## Overview

The unified test framework:

1. Automatically detects and loads all test prompts from the `prompts` directory
2. Tests each prompt against both Ollama API and the Claudacity proxy
3. Tests both streaming and non-streaming responses
4. Provides detailed reporting of test results
5. Identifies which configurations are working and which are failing

## Directory Structure

```
tests/
  ├── prompts/
  │    ├── 01_simple.txt      # Simple "Hello!" prompt
  │    ├── 02_complex.txt     # Complex distributed system prompt
  │    └── ...                # Add more prompts as needed
  │
  ├── unified_test_framework.py  # Main test framework
  ├── run_unified_tests.py       # Test runner script
  └── test_results/              # Directory for test results
       └── test_results_*.txt    # Generated test result files
```

## Adding New Test Prompts

To add a new test prompt, simply add a `.txt` file to the `prompts` directory. The framework will automatically discover and test it.

Naming convention:
- Use a numeric prefix to control the ordering: `01_`, `02_`, etc.
- Use a descriptive name that indicates what the prompt is testing
- Example: `03_code_generation.txt`

## Running Tests

### Basic Usage

```bash
# Run all tests
python tests/run_unified_tests.py

# Skip tests against the Ollama API (only test the proxy)
python tests/run_unified_tests.py --skip-ollama

# Skip streaming tests
python tests/run_unified_tests.py --skip-streaming

# Specify custom URLs
python tests/run_unified_tests.py --ollama-url http://localhost:11434/api/chat --proxy-url http://localhost:8000/api/chat
```

### Environment Variables

You can also configure the test parameters using environment variables:

```bash
# Set the Ollama URL
export OLLAMA_URL="http://localhost:11434/api/chat"

# Set the Proxy URL
export PROXY_URL="http://localhost:8000/api/chat"

# Set the timeout
export TEST_TIMEOUT=600  # 10 minutes

# Run the tests
python tests/run_unified_tests.py
```

## Test Results

The test framework generates a detailed test report that includes:

1. Overall success rate
2. Success rate by endpoint and streaming mode
3. Success rate by prompt
4. Detailed list of failed tests with error messages
5. Timing information for each test

Example:

```
================================================================================
TEST SUMMARY: 7/8 tests passed (87.5%)
================================================================================

Results by Endpoint:
----------------------------------------
http://localhost:11434/api/chat - Streaming: 2/2 passed (100.0%)
http://localhost:11434/api/chat - Non-streaming: 2/2 passed (100.0%)
http://localhost:8000/api/chat - Streaming: 2/2 passed (100.0%)
http://localhost:8000/api/chat - Non-streaming: 1/2 passed (50.0%)

Results by Prompt:
----------------------------------------
01_simple.txt: 4/4 passed (100.0%)
02_complex.txt: 3/4 passed (75.0%)

Failed Tests:
----------------------------------------
FAILURE: http://localhost:8000/api/chat - Non-streaming - 02_complex.txt (timeout after 300.0s) - Error: Timeout after 300.00s
```

## Integration with CI/CD

The test scripts are designed to work in CI/CD environments:

- Returns exit code 0 if all tests pass, 1 if any fail
- Generates timestamped test result files
- Supports configuration via environment variables
- Works with Jenkins, GitHub Actions, and other CI systems