import sys

def read_route():
    print("\n--- Route Header ---")
    try:
        with open("/usr/local/lib/python3.11/dist-packages/airflow/api_fastapi/execution_api/routes/task_instances.py", "r") as f:
            lines = f.readlines()
            for i in range(0, 105):
                if i < len(lines):
                    print(f"{i+1}: {lines[i].rstrip()}")
    except Exception as e:
        print(f"Error checking route: {e}")

if __name__ == "__main__":
    read_route()
