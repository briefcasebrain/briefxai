#!/usr/bin/env python3
"""
Complete test suite for BriefX Python implementation
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
    from briefx.monitoring import monitoring_system
    from briefx.error_recovery import error_recovery_system

    examples = generate_example_conversations(count=3, seed=42)
    assert len(examples) == 3

    preprocessor = SmartPreprocessor()
    processed, report = preprocessor.preprocess(examples)
    assert report.overall_quality_score > 0

    monitoring_system.record_request(True, 0.5)
    metrics = monitoring_system.get_metrics()
    assert metrics['total_requests'] >= 1


def test_api_endpoints():
    """Test Flask API endpoints"""
    from app import app

    client = app.test_client()

    response = client.get('/api/monitoring/health')
    assert response.status_code == 200
    health_data = json.loads(response.data)
    assert health_data['data']['status'] == 'healthy'

    response = client.get('/api/monitoring/metrics')
    assert response.status_code == 200
    metrics_data = json.loads(response.data)
    assert metrics_data['success'] is True

    response = client.post('/api/generate-examples',
                           json={'count': 2, 'seed': 42})
    assert response.status_code == 200
    example_data = json.loads(response.data)
    assert example_data['data']['count'] == 2

    response = client.get('/api/example')
    assert response.status_code == 200
    example = json.loads(response.data)
    assert 'conversations' in example


def test_cli_commands():
    """Test CLI interface"""
    from cli import cli

    expected_commands = ['ui', 'analyze', 'example', 'serve']
    registered = list(cli.commands.keys())

    for cmd in expected_commands:
        assert cmd in registered, f"Command '{cmd}' not registered"


def test_data_models():
    """Test data models"""
    from briefx.data.models import ConversationData, Message

    msg = Message(role="user", content="Test message")
    assert msg.role == "user"
    assert msg.content == "Test message"

    conv = ConversationData(
        messages=[msg],
        metadata={"test": True}
    )
    assert len(conv.messages) == 1
    assert conv.metadata["test"] is True


def test_providers():
    """Test provider system"""
    from briefx.providers.factory import get_available_providers

    providers = get_available_providers()
    assert len(providers) > 0

    for name, info in providers.items():
        assert 'name' in info
        assert 'available' in info
