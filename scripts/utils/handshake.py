# AgentQMS/tools/debug/handshake_test.py
import redis
import requests
import json
import os

def test_infrastructure():
    print("🚀 Starting AgentQMS Infrastructure Handshake...")
    
    # 1. Test Redis (The Virtual State Store)
    print("\n[1/2] Checking Redis...")
    try:
        r = redis.Redis(host='redis', port=6379, socket_timeout=5)
        r.set('handshake_test', 'success')
        val = r.get('handshake_test').decode('utf-8')
        print(f"✅ Redis Reachable: {val}")
    except Exception as e:
        print(f"❌ Redis Failed: {e}")

    # 2. Test Ollama (The Windows Host LLM)
    print("\n[2/2] Checking Ollama (Windows Host)...")
    ollama_url = "http://host.docker.internal:11434/api/tags"
    try:
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"✅ Ollama Reachable: Found {len(models)} models.")
            print(f"   Available: {', '.join(models[:3])}...")
        else:
            print(f"❌ Ollama returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Ollama Failed: {e}")
        print("   Tip: Ensure OLLAMA_HOST=0.0.0.0 and Firewall rule is active.")

if __name__ == "__main__":
    test_infrastructure()