#!/usr/bin/env python3
"""
Test connection to Airflow API.
Diagnoses connectivity issues by attempting to connect to the Airflow API
using the internal Docker host IP (172.17.0.1) which is more reliable
than localhost in some container environments.
"""

import os
import requests
import sys

# Default to Docker gateway IP if not specified
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://172.17.0.1:8080")
USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

def test_connection():
    print(f"Testing connection to Airflow at: {AIRFLOW_URL}")
    print("-" * 50)

    if not _check_health():
        return False

    token = _check_auth()
    if token:
        _check_api_access(token)

    print("-" * 50)
    return True

def _check_health() -> bool:
    # 1. Test Basic Connectivity (Health/Public endpoint)
    try:
        # Try health endpoint first (Airflow 3 moved this)
        health_url = f"{AIRFLOW_URL}/api/v2/monitor/health"
        print(f"1. Checking Health ({health_url})...", end=" ")
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            print("✅ OK")
            print(f"   Status: {resp.json().get('metadatabase', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Failed (Status: {resp.status_code})")
            print(f"   Response: {resp.text[:100]}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("\nPossible causes:")
        print("- Airflow container is not running (docker ps)")
        print("- Port 8080 is not mapped or firewall blocked")
        print("- Wrong IP address (try localhost vs 172.17.0.1)")
        return False
    return True

def _check_auth() -> str | None:
    # 2. Test Authentication (Get Token)
    print("\n2. Testing Authentication...", end=" ")
    token_url = f"{AIRFLOW_URL}/auth/token"
    token = None

    try:
        # Airflow 3.x /auth/token
        headers = {"Content-Type": "application/json"}
        payload = {"username": USERNAME, "password": PASSWORD}

        resp = requests.post(token_url, json=payload, headers=headers, timeout=5)

        # Airflow 3 returns 201 Created for token
        if resp.status_code in [200, 201]:
            data = resp.json()
            token = data.get("access_token")
            if token:
                print("✅ OK (Token received)")
            else:
                print("❌ Failed (No token in response)")
        elif resp.status_code == 404:
             print("❌ Failed (404 Not Found - possibly wrong Airflow version or endpoint)")
        elif resp.status_code == 401 or resp.status_code == 403:
             print(f"❌ Failed (Auth Error: {resp.status_code}) - Check credentials")
        else:
            print(f"❌ Failed (Status: {resp.status_code})")
            try:
                print(f"   Response: {resp.text[:200]}")
            except Exception:
                pass
    except Exception as e:
        print(f"❌ Error: {e}")

    return token

def _check_api_access(token: str) -> None:
    # 3. Test API Access (List DAGs)
    print("\n3. Testing API Access (List DAGs)...", end=" ")
    api_url = f"{AIRFLOW_URL}/api/v2/dags"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            print("✅ OK")
            dags = resp.json().get("dags", [])
            print(f"   Found {len(dags)} DAGs:")
            for dag in dags[:5]:  # Show first 5
                print(f"   - {dag['dag_id']} ({'Paused' if dag['is_paused'] else 'Active'})")
            if len(dags) > 5:
                print("   ...")
        else:
            print(f"❌ Failed (Status: {resp.status_code})")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_connection()
