import sys
import os

def debug():
    try:
        with open("/usr/local/lib/python3.11/dist-packages/airflow/sdk/execution_time/supervisor.py", "r") as f:
            lines = f.readlines()
            # print lines 1910 to 1940
            for i in range(1910, 1940):
                if i < len(lines):
                    print(f"{i+1}: {lines[i].rstrip()}")
    except Exception as e:
        print(f"Error reading source: {e}")

if __name__ == "__main__":
    debug()
