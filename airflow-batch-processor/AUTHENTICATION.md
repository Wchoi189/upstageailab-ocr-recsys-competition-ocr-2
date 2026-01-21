# Airflow API Authentication Guide

This guide shows how to authenticate with the Airflow API running in Docker.

## Prerequisites

- Airflow 3.x running in Docker (accessible at http://localhost:8080 or http://172.17.0.1:8080)
- Default admin credentials: `admin` / `admin`
- **Note**: Airflow 3.x uses `/api/v2` endpoints (v1 is deprecated) and `/auth/token` for JWT generation

## Method 1: JWT Token (Airflow 3.x Native)

Airflow 3.x provides a built-in JWT token endpoint:

```bash
# Generate JWT token (adjust IP if needed: localhost or 172.17.0.1)
curl -X POST "http://172.17.0.1:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Response:
# {"access_token":"eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9..."}

# Extract and use the token
TOKEN=$(curl -s -X POST "http://172.17.0.1:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' | jq -r '.access_token')

# Use the token for API calls
curl -X GET "http://172.17.0.1:8080/api/v2/dags" \
  -H "Authorization: Bearer $TOKEN"
```

## Method 2: Basic Authentication (Simple Alternative)

The simplest and most commonly used method with Airflow:

```bash
# List all DAGs
curl -X GET "http://localhost:8080/api/v2/dags" \
  --user "admin:admin"

# Get specific DAG info
curl -X GET "http://localhost:8080/api/v2/dags/batch_processor_dag" \
  --user "admin:admin"

# Trigger a DAG run
curl -X POST "http://localhost:8080/api/v2/dags/batch_processor_dag/dagRuns" \
  --user "admin:admin" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Method 3: Create API User Inside Container

```bash
# Create a dedicated API user
docker exec airflow-webserver airflow users create \
  --username api_user \
  --firstname API \
  --lastname User \
  --email api@example.com \
  --role Admin \
  --password "your_secure_password_here"

# Use the new credentials
curl -X GET "http://localhost:8080/api/v2/dags" \
  --user "api_user:your_secure_password_here"
```

## Method 4: Generate Custom JWT Token (Advanced)

If you need custom JWT for non-standard authentication (usually not needed with Airflow 3.x):

### Install dependencies from WSL2

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/airflow-batch-processor/scripts
pip install -r requirements.txt
```

### Generate token

```bash
# Set your secret key (should match Airflow's secret_key config)
export AIRFLOW_SECRET_KEY="your-airflow-secret-key"
export AIRFLOW_USERNAME="admin"
export JWT_EXPIRATION_HOURS="24"

python generate_jwt.py
```

### Use the token

```bash
# Copy the generated token
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

# Use it in API requests
curl -X GET "http://localhost:8080/api/v2/dags" \
  -H "Authorization: Bearer $TOKEN"
```

## Method 5: Environment Variables for Scripts

Save credentials in a `.env` file for scripts:

```bash
# In airflow-batch-processor/.env
AIRFLOW_API_URL=http://localhost:8080/api/v2
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
```

Then use in Python:

```python
import os
import requests
from requests.auth import HTTPBasicAuth

url = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v2")
username = os.getenv("AIRFLOW_USERNAME", "admin")
password = os.getenv("AIRFLOW_PASSWORD", "admin")

response = requests.get(
    f"{url}/dags",
    auth=HTTPBasicAuth(username, password)
)
print(response.json())
```

## Testing Your Authentication

```bash
# Test connection (should return 200 OK with DAG list)
curl -v -X GET "http://localhost:8080/api/v2/dags" \
  --user "admin:admin"

# Check Airflow version
curl -X GET "http://localhost:8080/api/v2/version" \
  --user "admin:admin"

# Get connection info
curl -X GET "http://localhost:8080/api/v2/connections" \
  --user "admin:admin"
```

## Quick Token Helper Script

```bash
# Save as get_airflow_token.sh
#!/bin/bash
AIRFLOW_URL="${AIRFLOW_URL:-http://172.17.0.1:8080}"
USERNAME="${AIRFLOW_USERNAME:-admin}"
PASSWORD="${AIRFLOW_PASSWORD:-admin}"

echo "Getting token from $AIRFLOW_URL..."
TOKEN=$(curl -s -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}" | jq -r '.access_token')

if [ "$TOKEN" = "null" ] || [ -z "$TOKEN" ]; then
  echo "Failed to get token. Check your credentials and URL."
  exit 1
fi

echo "Token obtained successfully:"
echo "$TOKEN"
echo ""
echo "Export to environment:"
echo "export AIRFLOW_TOKEN=$TOKEN"
```

## Important Notes

1. **Airflow 3.x Changes**: The webserver command is now `airflow api-server`, JWT endpoint is `/auth/token`
2. **Default Credentials**: Change default `admin:admin` in production
3. **Secret Key**: For JWT, ensure your secret key matches Airflow's `secret_key` configuration
4. **RBAC**: Airflow uses role-based access control; ensure users have appropriate roles

## Troubleshooting

### Connection Refused
```bash
# Check if Airflow is running
docker ps | grep airflow-webserver

# Check logs
docker logs airflow-webserver
```

### Authentication Failed
```bash
# Reset admin password inside container
docker exec -it airflow-webserver bash
airflow users create --username admin --firstname Air --lastname Flow \
  --email admin@example.com --role Admin --password newpassword
```

### Port Not Accessible
```bash
# Verify port mapping
docker port airflow-webserver 8080
```
