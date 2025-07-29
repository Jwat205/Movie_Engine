#!/usr/bin/env python3
"""
Simple test script for the Movie Recommendation API
Run this while your server is running to test the endpoints directly
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    print("🎬 Testing Movie Recommendation API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {health['status']}")
            print(f"   Models loaded: {health.get('models_loaded', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            root = response.json()
            print("✅ Root endpoint working!")
            print(f"   Message: {root['message']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Get Recommendations
    print("\n3. Testing Recommendations...")
    test_cases = [
        {"user_id": 1, "method": "hybrid", "num_recommendations": 5},
        {"user_id": 10, "method": "collaborative", "num_recommendations": 3},
        {"user_id": 50, "method": "content", "num_recommendations": 7},
        {"user_id": 100, "method": "svd", "num_recommendations": 10}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test 3.{i}: {test_case['method']} method for user {test_case['user_id']}")
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/recommendations",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Got {len(data.get('recommendations', []))} recommendations")
                print(f"   ⚡ Response time: {(end_time - start_time)*1000:.2f}ms")
                
                # Show first few recommendations
                recs = data.get('recommendations', [])
                for j, rec in enumerate(recs[:3], 1):
                    if isinstance(rec, dict):
                        title = rec.get('title', 'Unknown')
                        genre = rec.get('genre', 'Unknown')
                        score = rec.get('score', 0)
                        print(f"      {j}. {title} ({genre}) - Score: {score:.3f}")
                    else:
                        print(f"      {j}. {rec}")
                        
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 4: User Profile
    print("\n4. Testing User Profile...")
    try:
        response = requests.get(f"{BASE_URL}/user/1/profile")
        if response.status_code == 200:
            profile = response.json()
            print("✅ User profile retrieved!")
            print(f"   User interactions: {profile.get('total_interactions', 0)}")
            print(f"   Avg watch %: {profile.get('avg_watch_percentage', 0):.1f}%")
        elif response.status_code == 404:
            print("⚠️ User not found (expected for some users)")
        else:
            print(f"❌ Profile failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Profile error: {e}")
    
    # Test 5: Metrics
    print("\n5. Testing Metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Metrics retrieved!")
            print("   Model Performance:")
            for model, perf in metrics.get('model_performance', {}).items():
                rmse = perf.get('rmse', 0)
                precision = perf.get('precision_at_10', 0)
                print(f"      {model.upper()}: RMSE={rmse:.4f}, P@10={precision:.4f}")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")

if __name__ == "__main__":
    test_api()