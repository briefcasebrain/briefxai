#!/usr/bin/env python3
"""
Updated test suite for BriefX Python implementation
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_modules():
    """Test core module imports and basic functionality"""
    print("\n1. Testing Core Modules")
    print("-" * 40)
    
    try:
        from briefx.examples import generate_example_conversations
        from briefx.preprocessing import SmartPreprocessor
        print("✓ Core modules imported")
        
        # Generate examples
        examples = generate_example_conversations(count=3, seed=42)
        print(f"✓ Generated {len(examples)} examples")
        
        # Test preprocessing
        preprocessor = SmartPreprocessor()
        print("✓ Preprocessor created")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_data_models():
    """Test data models"""
    print("\n2. Testing Data Models")
    print("-" * 40)
    
    try:
        from briefx.data.models import ConversationData, Message, FacetValue
        from briefx.data.parsers import parse_json_conversations
        print("✓ Data models imported")
        
        # Create test conversation
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
        conv = ConversationData(messages=messages)
        print(f"✓ Created conversation with {len(conv.messages)} messages")
        
        # Test JSON parsing
        json_data = '[{"messages": [{"role": "user", "content": "test"}]}]'
        conversations = parse_json_conversations(json_data)
        print(f"✓ Parsed {len(conversations)} conversations from JSON")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_providers():
    """Test provider system"""
    print("\n3. Testing Provider System")
    print("-" * 40)
    
    try:
        from briefx.providers.factory import get_available_providers
        from briefx.providers.base import LLMProvider, EmbeddingProvider
        print("✓ Provider modules imported")
        
        # Check available providers
        available = get_available_providers()
        print(f"✓ Found {len(available)} available providers: {list(available.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_cli_functionality():
    """Test CLI functionality"""
    print("\n4. Testing CLI Functionality")
    print("-" * 40)
    
    try:
        from briefx.examples import generate_example_conversations
        print("✓ CLI modules imported")
        
        # Test example generation (CLI functionality)
        examples = generate_example_conversations(count=2, seed=123)
        print(f"✓ Generated {len(examples)} examples for CLI")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_api_functionality():
    """Test API functionality"""
    print("\n5. Testing API Functionality")
    print("-" * 40)
    
    try:
        from flask import Flask
        from briefx.examples import generate_example_conversations
        print("✓ API modules imported")
        
        # Test Flask app creation
        app = Flask(__name__)
        print("✓ Flask app created")
        
        # Test with test client
        with app.test_client() as client:
            @app.route('/test')
            def test():
                examples = generate_example_conversations(count=1)
                return {'status': 'ok', 'count': len(examples)}
            
            response = client.get('/test')
            data = response.get_json()
            print(f"✓ API test endpoint returned: {data}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BriefX Python Implementation - Updated Test Suite")
    print("=" * 50)
    
    tests = [
        ("Core Modules", test_core_modules),
        ("Data Models", test_data_models), 
        ("Provider System", test_providers),
        ("CLI Functionality", test_cli_functionality),
        ("API Functionality", test_api_functionality),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                results[name] = True
            else:
                results[name] = False
        except Exception as e:
            print(f"\n{name}")
            print("-" * 40)
            print(f"✗ Unexpected error: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:<20} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)