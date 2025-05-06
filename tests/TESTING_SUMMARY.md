# Claudacity Testing Summary

This document summarizes the testing tools and frameworks available for the Claudacity project.

## 1. Unified Test Framework

The Unified Test Framework provides comprehensive testing of both the Ollama API and Claudacity proxy server using multiple prompts and response modes.

### Key Features:
- Tests both streaming and non-streaming responses
- Tests multiple prompts (simple, complex, etc.)
- Automatically discovers prompts from the `prompts` directory
- Provides detailed test results and reports
- Configurable via command line or environment variables

### Usage:
```bash
# Run all tests
python tests/run_unified_tests.py

# Run with specific options
python tests/run_unified_tests.py --skip-ollama --timeout 600
```

See [UNIFIED_TEST_FRAMEWORK.md](UNIFIED_TEST_FRAMEWORK.md) for detailed documentation.

## 2. Timeout Verification

The Timeout Verification tool specifically tests the timeout handling and process termination mechanism for hung Claude processes.

### Key Features:
- Verifies that timeout detection works correctly
- Confirms that hung processes are properly terminated
- Ensures terminated processes are untracked from the system
- Tests process-specific timeout configuration

### Usage:
```bash
# Run the timeout verification
python tests/verify_timeout_changes.py
```

See [TIMEOUT_INVESTIGATION_REPORT.md](TIMEOUT_INVESTIGATION_REPORT.md) for detailed findings.

## 3. Complex Prompt Testing

The unified test framework includes testing of the complex "distributed job scheduling system" prompt that was previously causing Claude to hang.

### Key Features:
- Tests both Ollama API and proxy with the complex prompt
- Measures timing between response chunks
- Provides detailed logs of chunk timing and delays
- Helps identify potential timeout issues

### Usage:
```bash
# Run only the complex prompt test
python tests/run_unified_tests.py --skip-ollama --skip-streaming
```

## 4. Individual Component Tests

The project includes tests for individual components:

- `test_api.py`: Tests the API endpoints
- `test_chat_endpoint.py`: Tests the chat endpoint functionality
- `test_dashboard.py`: Tests the dashboard
- `test_metrics.py`: Tests the metrics tracking system
- `test_ollama_compatibility.py`: Tests compatibility with Ollama

## Test Files Organization

The test files are organized as follows:

```
tests/
  ├── prompts/                  # Test prompts
  │    ├── 01_simple.txt
  │    ├── 02_complex.txt
  │    └── ...
  │
  ├── test_results/             # Test results directory
  │
  ├── unified_test_framework.py # Main test framework
  ├── run_unified_tests.py      # Test runner script
  ├── verify_timeout_changes.py # Timeout verification
  │
  ├── UNIFIED_TEST_FRAMEWORK.md # Framework documentation
  ├── TIMEOUT_INVESTIGATION_REPORT.md # Timeout investigation report
  └── TESTING_SUMMARY.md        # This file
```

## Adding New Tests

1. **Adding new prompts**: Add a new `.txt` file to the `prompts` directory.
2. **Adding new test cases**: Create a new test file in the `tests` directory.
3. **Extending the framework**: Modify `unified_test_framework.py` to add new test types.