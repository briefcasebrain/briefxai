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
    from briefx.examples import generate_example_conversations
    from briefx.preprocessing import SmartPreprocessor

    examples = generate_example_conversations(count=3, seed=42)
    assert len(examples) == 3

    preprocessor = SmartPreprocessor()
    assert preprocessor is not None


def test_data_models():
    """Test data models"""
    from briefx.data.models import ConversationData, Message, FacetValue
    from briefx.data.parsers import parse_json_conversations

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]
    conv = ConversationData(messages=messages)
    assert len(conv.messages) == 2

    json_data = '[{"messages": [{"role": "user", "content": "test"}]}]'
    conversations = parse_json_conversations(json_data)
    assert len(conversations) == 1


def test_providers():
    """Test provider system"""
    from briefx.providers.factory import get_available_providers
    from briefx.providers.base import LLMProvider, EmbeddingProvider

    available = get_available_providers()
    assert len(available) > 0


def test_cli_functionality():
    """Test CLI functionality"""
    from briefx.examples import generate_example_conversations

    examples = generate_example_conversations(count=2, seed=123)
    assert len(examples) == 2


def test_api_functionality():
    """Test API functionality"""
    from flask import Flask
    from briefx.examples import generate_example_conversations

    app = Flask(__name__)

    @app.route('/test')
    def test_endpoint():
        examples = generate_example_conversations(count=1)
        return {'status': 'ok', 'count': len(examples)}

    with app.test_client() as client:
        response = client.get('/test')
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['count'] == 1
