#!/usr/bin/env python3
"""Generate JWT token for Airflow API authentication."""

import jwt
import datetime
import os

# Configuration
SECRET_KEY = os.getenv("AIRFLOW_SECRET_KEY", "your-secret-key-change-this")
USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

def generate_jwt_token(username: str, expiration_hours: int = 24) -> str:
    """
    Generate a JWT token for Airflow API authentication.

    Args:
        username: The username to encode in the token
        expiration_hours: Token expiration time in hours

    Returns:
        JWT token string
    """
    payload = {
        "sub": username,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=expiration_hours),
        "username": username,
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def main():
    """Generate and display JWT token."""
    print("=" * 60)
    print("Airflow JWT Token Generator")
    print("=" * 60)
    print(f"Username: {USERNAME}")
    print(f"Expiration: {EXPIRATION_HOURS} hours")
    print(f"Secret Key: {SECRET_KEY[:10]}... (truncated)")
    print("=" * 60)

    token = generate_jwt_token(USERNAME, EXPIRATION_HOURS)

    print("\nGenerated JWT Token:")
    print("-" * 60)
    print(token)
    print("-" * 60)

    print("\nUsage Example:")
    print('curl -X GET "http://localhost:8080/api/v2/dags" \\')
    print(f'  -H "Authorization: Bearer {token}"')
    print("\nOr with Basic Auth (simpler):")
    print('curl -X GET "http://localhost:8080/api/v2/dags" \\')
    print(f'  --user "{USERNAME}:your_password"')


if __name__ == "__main__":
    main()
