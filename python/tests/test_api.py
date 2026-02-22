#!/usr/bin/env python3
"""
Test BriefX API endpoints
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from briefx.examples import generate_example_conversations

def test_api_endpoints():
    """Test key API endpoints"""
    
    # Create test client
    client = app.test_client()
    
    print("Testing BriefX API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    response = client.get('/api/monitoring/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    print(f"✓ Health check: {data['data']['status']}")
    
    # Test 2: Metrics
    response = client.get('/api/monitoring/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    print(f"✓ Metrics: {data['data']['total_requests']} requests")
    
    # Test 3: Generate examples via API
    response = client.post('/api/generate-examples', 
                          json={'count': 3, 'seed': 42})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    assert data['data']['count'] == 3
    print(f"✓ Example generation: {data['data']['count']} conversations")
    
    # Test 4: Preprocess data
    test_conversations = generate_example_conversations(count=2, seed=42)
    conv_data = [{'messages': [{'role': m.role, 'content': m.content} for m in conv.messages], 'metadata': conv.metadata} for conv in test_conversations]
    
    response = client.post('/api/preprocess',
                          json={'conversations': conv_data})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    quality = data['data']['quality_report']['overall_quality_score']
    print(f"✓ Preprocessing: quality score {quality:.2%}")
    
    # Test 5: Session status endpoint (non-existent session returns not_found)
    response = client.get('/api/session/test-session/status')
    assert response.status_code == 200
    data = json.loads(response.data)
    print(f"✓ Status check: {data['status']}")
    
    # Test 6: Example data endpoint
    response = client.get('/api/example')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'conversations' in data
    assert 'clusters' in data
    print(f"✓ Example data: {len(data['conversations'])} conversations, {len(data['clusters'])} clusters")
    
    # Test 7: Upload endpoint (simulate)
    response = client.post('/api/upload', data={})
    data = json.loads(response.data)
    assert data['success'] == False  # Should fail with no files
    print(f"✓ Upload endpoint: working (empty test)")
    
    # Test 8: Providers info
    response = client.get('/api/providers')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'providers' in data
    print(f"✓ Providers: {len(data['providers'])} available")
    
    print("\n" + "=" * 50)
    print("🎉 All API tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_api_endpoints()