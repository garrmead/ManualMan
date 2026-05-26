FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/uploads data/extracted_images data/chroma_db

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD streamlit run app.py --server.port=${PORT:-8501}
