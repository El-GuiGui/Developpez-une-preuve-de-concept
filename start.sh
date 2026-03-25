#!/usr/bin/env bash
set -e

#export API_BASE="http://localhost:8000"

# API (backend on localhost:8000)
#python -m uvicorn api.main:app --host localhost --port 8000 &

pip install -r requirements.txt

# Streamlit (frontend on 0.0.0.0:5000)

streamlit run app_dashboard/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 5000 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
