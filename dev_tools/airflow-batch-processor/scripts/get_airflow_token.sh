#!/bin/bash
# Get JWT token from Airflow 3.x using the native /auth/token endpoint

set -e

AIRFLOW_URL="${AIRFLOW_URL:-http://172.17.0.1:8080}"
USERNAME="${AIRFLOW_USERNAME:-admin}"
PASSWORD="${AIRFLOW_PASSWORD:-admin}"

echo "Getting token from $AIRFLOW_URL..."

RESPONSE=$(curl -s -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

TOKEN=$(echo "$RESPONSE" | jq -r '.access_token')

if [ "$TOKEN" = "null" ] || [ -z "$TOKEN" ]; then
  echo "❌ Failed to get token. Check your credentials and URL."
  echo "Response: $RESPONSE"
  exit 1
fi

echo "✅ Token obtained successfully!"
echo ""
echo "Token:"
echo "$TOKEN"
echo ""
echo "Export to environment:"
echo "export AIRFLOW_TOKEN='$TOKEN'"
echo ""
echo "Test the token:"
echo "curl -X GET \"$AIRFLOW_URL/api/v2/dags\" -H \"Authorization: Bearer \$AIRFLOW_TOKEN\""
