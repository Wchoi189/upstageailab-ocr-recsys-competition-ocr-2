import sys
import os

TARGET_FILE = "/usr/local/lib/python3.11/dist-packages/airflow/sdk/execution_time/supervisor.py"

def inject():
    try:
        with open(TARGET_FILE, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        injected = False
        in_supervise = False
        
        for line in lines:
            if "def supervise" in line:
                in_supervise = True
                print(f"DEBUG match function: {line.strip()}")
            
            if in_supervise and "if" in line and "server" in line:
                 print(f"DEBUG match candidate: {line.strip()}")
                 if "if not server" in line and not injected:
                      new_lines.append('        print(f"DEBUG SUPERVISE: server={server} client={client}")\n')
                      print("INJECTED!")
                      injected = True
                      # Do not consume the line, append it after
            
            new_lines.append(line)

        if not injected:
            print("Could not find injection point.")
            # Don't write back if failed
            return

        with open(TARGET_FILE, "w") as f:
            f.writelines(new_lines)
            
        print("File updated successfully.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inject()
