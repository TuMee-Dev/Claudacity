"""
Test Ollama API compatibility with Claude service.
This script tests the Ollama-compatible endpoints provided by the Claude service.
"""

import json
import aiohttp
import asyncio
import argparse
import sys
import time
from typing import Dict, List, Optional, Union

# Default URLs
CLAUDE_URL = "http://localhost:22434"
OLLAMA_URL = "http://localhost:11434"

async def test_version_endpoint(base_url: str) -> Dict:
    """Test the /api/version endpoint"""
    url = f"{base_url}/api/version"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error: {response.status} - {error_text}")
                return {"error": response.status}
            
            return await response.json()

async def test_tags_endpoint(base_url: str) -> Dict:
    """Test the /api/tags endpoint"""
    url = f"{base_url}/api/tags"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error: {response.status} - {error_text}")
                return {"error": response.status}
            
            return await response.json()

async def test_chat_streaming(base_url: str, model: str, prompt: str) -> None:
    """Test the streaming /api/chat endpoint"""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    print(f"\n--- Streaming Chat Test ({base_url}) ---")
    print(f"Prompt: '{prompt}'")
    print("Response:")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status} - {error_text}")
                    return
                
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            chunk = json.loads(line)
                            if chunk.get("message", {}).get("content"):
                                content = chunk["message"]["content"]
                                full_response += content
                                print(content, end="", flush=True)
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {line}")
                
                print("\n\nFull response:", full_response)
                
    except Exception as e:
        print(f"Exception: {e}")

async def test_chat_non_streaming(base_url: str, model: str, prompt: str) -> Dict:
    """Test the non-streaming /api/chat endpoint"""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    print(f"\n--- Non-Streaming Chat Test ({base_url}) ---")
    print(f"Prompt: '{prompt}'")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status} - {error_text}")
                    return {"error": response.status}
                
                result = await response.json()
                print(f"Response: '{result.get('message', {}).get('content', '')}'")
                return result
                
    except Exception as e:
        print(f"Exception: {e}")
        return {"error": str(e)}

async def compare_endpoints(claude_url: str, ollama_url: str) -> None:
    """Compare endpoints between Claude and Ollama services"""
    print("\n=== Comparing /api/version Endpoints ===")
    claude_version = await test_version_endpoint(claude_url)
    ollama_version = await test_version_endpoint(ollama_url)
    
    print(f"Claude version: {json.dumps(claude_version, indent=2)}")
    print(f"Ollama version: {json.dumps(ollama_version, indent=2)}")
    
    print("\n=== Comparing /api/tags Endpoints ===")
    claude_tags = await test_tags_endpoint(claude_url)
    ollama_tags = await test_tags_endpoint(ollama_url)
    
    # Validate that the endpoints match the expected format
    print("Validating endpoint formats...")
    
    # Claude version endpoint should have a version field
    assert "version" in claude_version, "Claude version endpoint missing 'version' field"
    print("✓ Claude version endpoint has the correct format")
    
    # Ollama version endpoint should have a version field
    assert "version" in ollama_version, "Ollama version endpoint missing 'version' field"
    print("✓ Ollama version endpoint has the correct format")
    
    # Claude tags endpoint should have a models array
    assert "models" in claude_tags, "Claude tags endpoint missing 'models' field"
    assert len(claude_tags["models"]) > 0, "Claude tags endpoint has empty models array"
    print("✓ Claude tags endpoint has the correct format")
    
    # Ollama tags endpoint should have a models array
    assert "models" in ollama_tags, "Ollama tags endpoint missing 'models' field"
    assert len(ollama_tags["models"]) > 0, "Ollama tags endpoint has empty models array"
    print("✓ Ollama tags endpoint has the correct format")
    
    # Check required fields in model objects
    for service_name, tags, url in [("Claude", claude_tags, claude_url), ("Ollama", ollama_tags, ollama_url)]:
        for model in tags["models"]:
            required_fields = ["name", "model", "modified_at", "size", "digest", "details"]
            for field in required_fields:
                assert field in model, f"{service_name} model missing '{field}' field: {json.dumps(model)}"
            
            detail_fields = ["family", "parameter_size", "quantization_level"]
            for field in detail_fields:
                assert field in model["details"], f"{service_name} model details missing '{field}' field: {json.dumps(model['details'])}"
    
    print("✓ Both Claude and Ollama model objects have the required fields")
    
    # Just print the first model from each to avoid flooding the console
    print(f"Claude first model: {json.dumps(claude_tags.get('models', [])[0] if claude_tags.get('models') else {}, indent=2)}")
    print(f"Ollama first model: {json.dumps(ollama_tags.get('models', [])[0] if ollama_tags.get('models') else {}, indent=2)}")
    
    print("\n=== Testing Live Chat Endpoints ===")
    print("NOTE: This will test the actual running services at their respective URLs")
    print(f"Claude service: {claude_url}")
    print(f"Ollama service: {ollama_url}")
    
    # For Claude use the first available model
    claude_model = claude_tags.get('models', [])[0]['name'] if claude_tags.get('models') else "claude-3.7-sonnet"
    # For Ollama use the specified model or a default one
    ollama_model = ollama_tags.get('models', [])[0]['name'] if ollama_tags.get('models') else "microsoft/phi-4:latest"
    
    test_prompt = "Tell me a short joke about programming."
    
    print(f"\nUsing Claude model: {claude_model}")
    print(f"Using Ollama model: {ollama_model}")
    
    try:
        # Test non-streaming first (easier to compare)
        print("\n--- Testing Non-streaming Chat ---")
        claude_result = await test_chat_non_streaming(claude_url, claude_model, test_prompt)
        ollama_result = await test_chat_non_streaming(ollama_url, ollama_model, test_prompt)
        
        # Validate response format
        for service_name, result in [("Claude", claude_result), ("Ollama", ollama_result)]:
            required_fields = ["model", "message", "done"]
            for field in required_fields:
                if field not in result:
                    print(f"⚠️ {service_name} response missing '{field}' field")
            
            if "message" in result and isinstance(result["message"], dict):
                message = result["message"]
                if "role" not in message:
                    print(f"⚠️ {service_name} message missing 'role' field")
                if "content" not in message:
                    print(f"⚠️ {service_name} message missing 'content' field")
        
        # Test streaming (outputs to console)
        print("\n--- Testing Streaming Chat ---")
        await test_chat_streaming(claude_url, claude_model, test_prompt)
        await test_chat_streaming(ollama_url, ollama_model, test_prompt)
        
    except Exception as e:
        print(f"Error during chat testing: {str(e)}")

async def run_tests(args):
    """Run the selected tests based on command line args"""
    if args.compare:
        await compare_endpoints(args.claude_url, args.ollama_url)
        return
    
    if args.test_type in ["version", "all"]:
        print("\n=== Testing /api/version ===")
        result = await test_version_endpoint(args.claude_url)
        print(json.dumps(result, indent=2))
    
    if args.test_type in ["tags", "all"]:
        print("\n=== Testing /api/tags ===")
        result = await test_tags_endpoint(args.claude_url)
        print(json.dumps(result, indent=2))
    
    if args.test_type in ["chat", "all"]:
        model = "claude-3.7-sonnet"
        prompt = "Tell me a short joke about programming."
        
        # Test non-streaming chat
        print("\n=== Testing Non-streaming Chat ===")
        await test_chat_non_streaming(args.claude_url, model, prompt)
        
        # Test streaming chat
        print("\n=== Testing Streaming Chat ===")
        await test_chat_streaming(args.claude_url, model, prompt)

def main():
    parser = argparse.ArgumentParser(description="Test Ollama API compatibility with Claude service")
    parser.add_argument("--claude-url", default=CLAUDE_URL, help=f"Claude service URL (default: {CLAUDE_URL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"Ollama service URL (default: {OLLAMA_URL})")
    parser.add_argument("--test-type", choices=["version", "tags", "chat", "all"], default="all", 
                        help="Type of test to run (default: all)")
    parser.add_argument("--compare", action="store_true", help="Compare Claude and Ollama endpoints")
    
    args = parser.parse_args()
    
    # Run the async test function inside an event loop
    asyncio.run(run_tests(args))

if __name__ == "__main__":
    main()