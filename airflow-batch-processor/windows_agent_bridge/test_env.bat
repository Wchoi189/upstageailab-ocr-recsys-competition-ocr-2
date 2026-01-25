@echo off 
call .venv\Scripts\activate.bat 
set RABBITMQ_HOST=localhost 
python bridge_client.py "docker ps" 
echo Test completed. 
pause 
