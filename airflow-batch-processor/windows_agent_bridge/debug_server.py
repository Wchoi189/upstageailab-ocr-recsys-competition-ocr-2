import subprocess
import sys

# Test if we can run the server and see its output
result = subprocess.run([sys.executable, 'bridge_server.py'], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        timeout=10)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)