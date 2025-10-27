FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all code: api, ui, exported_model, etc
COPY . .

EXPOSE 9000
EXPOSE 8501

# Entry: start FastAPI + Streamlit
CMD uvicorn predictapi.api:app --host 0.0.0.0 --port 9000 & streamlit run ui/streamlit_app.py --server.port 8501 --server.headless true
