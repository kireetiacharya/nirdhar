# Nirdhar valuation engine — container for Hugging Face Spaces (Docker SDK), Render, Railway or Fly.io.
# The model is rebuilt inside the image (about a minute), so the saved engine always matches
# the installed library versions.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# data/real/*.csv (OpenStreetMap-derived Pune layers) ship with the repo;
# synthetic transactions are generated and the model is trained at build time.
RUN python -m nirdhar.generate && python -W ignore -m nirdhar.train

EXPOSE 7860
CMD ["sh", "-c", "python -W ignore -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
