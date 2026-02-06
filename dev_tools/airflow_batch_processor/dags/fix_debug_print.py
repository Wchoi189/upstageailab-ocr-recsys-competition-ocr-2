import sys
import os

TARGET_FILE = "/usr/local/lib/python3.11/dist-packages/airflow/sdk/execution_time/supervisor.py"

def fix():
    try:
        with open(TARGET_FILE, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        fixed = False
        
        for line in lines:
            if 'print(f"DEBUG SUPERVISE:' in line:
                if not line.startswith("            "):
                    # Fix indentation to 12 spaces
                    new_line = '            ' + line.lstrip()
                    new_lines.append(new_line)
                    fixed = True
                    print("Fixed indentation.")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if not fixed:
            print("No fix needed or line not found.")
            # return

        with open(TARGET_FILE, "w") as f:
            f.writelines(new_lines)
            
        print("File updated successfully.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix()
