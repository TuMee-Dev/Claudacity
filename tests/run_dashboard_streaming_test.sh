#!/bin/bash
# Script to test the dashboard streaming display

# Set the API server URL
SERVER_URL="http://localhost:22434"

# Make sure server is running
echo "Checking if server is running at $SERVER_URL..."
curl -s "$SERVER_URL/status" || {
    echo "ERROR: Server not running at $SERVER_URL"
    echo "Please start the server with: python claude_service.py --start"
    exit 1
}

# Run the dashboard streaming test
echo "Running dashboard streaming test..."
python tests/test_dashboard_streaming.py --server-url $SERVER_URL

if [ $? -eq 0 ]; then
    echo "SUCCESS: Dashboard streaming test passed!"
    echo "Dashboard output has been saved to dashboard_streaming_test.html for inspection."
    echo "You can view this file in your browser to see how the streaming content appears."
else
    echo "FAILURE: Dashboard streaming test failed."
    echo "Check the log output above for details."
    echo "If the dashboard_streaming_test.html file was created, you can inspect it to see what went wrong."
fi