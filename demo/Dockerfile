FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY demo/requirements.txt /app/requirements.txt

# Install llama-cpp-python from pre-built wheel (avoids OOM during C++ compilation)
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time to a fixed path
RUN python -c "from huggingface_hub import hf_hub_download; p=hf_hub_download('Roopalgn/sahayak-tb-gemma4-e2b-gguf', 'gemma-4-e2b-it.Q4_K_M.gguf', local_dir='/app/models'); print(p)"
ENV SAHAYAK_MODEL_PATH=/app/models/gemma-4-e2b-it.Q4_K_M.gguf

# Copy inference code (postprocess + schema)
COPY inference/ /app/inference/
COPY data/ /app/data/

# Copy demo app and static files
COPY demo/app.py /app/app.py
COPY demo/static/ /app/static/

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run
CMD ["python", "app.py"]
