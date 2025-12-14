# veritas-backend

**Useful commands:**  
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\effnet-env\Scripts\Activate.ps1

To run the back-end server locally:
  > $env:PORT=5000; python -m waitress --port=5000 app:app