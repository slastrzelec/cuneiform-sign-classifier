# Demo image: the Streamlit app with the committed demo gallery.
# The trained checkpoint is NOT baked in: the app downloads it from the Hugging Face Hub on
# first start (see HF_REPO_ID in app.py), so the image builds from a fresh clone.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app.py .
COPY demo_samples/ demo_samples/

# Do not run as root.
RUN useradd --create-home app
USER app
ENV HF_HOME=/home/app/.cache/huggingface

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
