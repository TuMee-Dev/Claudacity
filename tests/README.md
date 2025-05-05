# Claude Ollama Proxy Test Suite

This directory contains automated tests for the Claude Ollama Proxy server. These tests help ensure the server runs reliably and can detect issues before deployment.

## Test Organization

The test suite is organized into three main categories:

1. **Unit Tests** - Test individual components in isolation
   - `test_metrics.py` - Tests for the metrics tracking system

2. **API Tests** - Test API endpoints using mocked Claude responses
   - `test_api.py` - Tests for chat completions and other API endpoints

3. **Dashboard Tests** - Test dashboard HTML generation
   - `test_dashboard.py` - Tests for dashboard templates and metrics display

## Running Tests

### Option 1: Run All Tests

To run all tests at once:

```bash
python tests/run_tests.py
```

For more detailed output:

```bash
python tests/run_tests.py --verbose
```

### Option 2: Run Specific Tests

Run only unit tests:

```bash
python tests/run_tests.py --unit
```

Run only API tests:

```bash
python tests/run_tests.py --api
```

Run only dashboard tests:

```bash
python tests/run_tests.py --dashboard
```

### Option 3: Run Quick Self-Test

The server includes a built-in self-test that can be run quickly without setting up a test environment:

```bash
python claude_ollama_server.py --test
```

## Test Dependencies

The tests require the following Python packages:
- unittest (built-in)
- fastapi.testclient
- pytest (recommended for more advanced test runs)

Install test dependencies:

```bash
pip install fastapi pytest
```

## Continuous Integration

For automated testing in CI environments, use:

```bash
python tests/run_tests.py --all
```

The script will return a non-zero exit code if any tests fail, compatible with most CI systems.

## Adding New Tests

When adding new features to the server:

1. Create unit tests in the appropriate test file
2. Add integration tests for any new endpoints
3. Run the full test suite to ensure everything still works

## Test Coverage

Current test coverage includes:
- Metrics tracking system
- API endpoints (chat completions, version, status)
- Dashboard HTML generation
- Conversation tracking
- Error handling

Future test additions:
- Stream response handling
- Process management
- OpenWebUI compatibility tests