#!/usr/bin/env python3
"""
Simple test script to verify BriefX Python implementation
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from briefx.examples import generate_example_conversations
from briefx.data.models import ConversationData
from briefx.preprocessing.smart_preprocessor import SmartPreprocessor
from briefx.preprocessing.validators import CompositeValidator
from briefx.monitoring import monitoring_system
from briefx.error_recovery import error_recovery_system


def test_data_models():
    """Test data models"""
    print("Testing data models...")
    
    # Create example conversation
    conversations = generate_example_conversations(count=3, seed=42)
    assert len(conversations) == 3
    assert all(isinstance(c, ConversationData) for c in conversations)
    assert all(len(c.messages) >= 4 for c in conversations)
    
    print("✓ Data models working")


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("Testing preprocessing...")
    
    # Generate test data
    conversations = generate_example_conversations(count=5, seed=42)
    
    # Create preprocessor
    preprocessor = SmartPreprocessor()
    
    # Analyze quality
    quality_report = preprocessor.analyze_data_quality(conversations)
    assert quality_report.total_conversations == 5
    assert quality_report.overall_quality_score > 0
    
    # Preprocess
    processed, report = preprocessor.preprocess(conversations)
    assert len(processed) <= len(conversations)
    
    print(f"✓ Preprocessing working (quality score: {report.overall_quality_score:.2%})")


def test_validation():
    """Test validation system"""
    print("Testing validation...")
    
    conversations = generate_example_conversations(count=3, seed=42)
    
    validator = CompositeValidator()
    result = validator.validate(conversations)
    
    assert result is not None
    assert hasattr(result, 'is_valid')
    assert hasattr(result, 'quality_score')
    
    print(f"✓ Validation working (quality: {result.quality_score:.2%})")


def test_monitoring():
    """Test monitoring system"""
    print("Testing monitoring...")
    
    # Record some metrics
    monitoring_system.record_request(success=True, duration=0.5)
    monitoring_system.record_api_call("openai", success=True, duration=1.2)
    monitoring_system.record_component_execution("analysis", duration=2.5, success=True)
    
    # Get metrics
    metrics = monitoring_system.get_metrics()
    assert metrics['total_requests'] == 1
    assert 'openai' in metrics['api_calls']
    
    # Perform health check
    health = monitoring_system.perform_health_check()
    assert health.status is not None
    
    print(f"✓ Monitoring working (health: {health.status.value})")


import pytest

@pytest.mark.asyncio
async def test_error_recovery():
    """Test error recovery system"""
    print("Testing error recovery...")
    
    # Test successful operation
    async def successful_op():
        return "success"
    
    result = await error_recovery_system.execute_with_recovery(
        "test_op",
        successful_op
    )
    assert result.success
    assert result.result == "success"
    
    # Test operation with retry
    call_count = 0
    
    async def failing_then_success():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success after retry"
    
    result = await error_recovery_system.execute_with_recovery(
        "api_call",
        failing_then_success
    )
    assert result.success
    assert result.attempts == 3
    
    print(f"✓ Error recovery working (retries: {result.attempts})")


def test_example_generation():
    """Test example generation"""
    print("Testing example generation...")
    
    # Test different categories
    for category in ["technical", "creative", "analytical"]:
        conversations = generate_example_conversations(
            count=2,
            category=category,
            seed=42
        )
        assert len(conversations) == 2
        assert all(c.metadata.get("category") == category for c in conversations)
    
    print("✓ Example generation working")


def main():
    """Run all tests"""
    print("=" * 50)
    print("BriefX Python Implementation Test Suite")
    print("=" * 50)
    
    try:
        # Run synchronous tests
        test_data_models()
        test_example_generation()
        test_preprocessing()
        test_validation()
        test_monitoring()
        
        # Run async tests
        asyncio.run(test_error_recovery())
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
        # Print summary
        print("\nSystem Summary:")
        print("- Data models: ✓")
        print("- Preprocessing: ✓")
        print("- Validation: ✓")
        print("- Monitoring: ✓")
        print("- Error recovery: ✓")
        print("- Example generation: ✓")
        
        print("\nThe Python implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()