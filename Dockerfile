# Serving image. Deliberately minimal, because Railway bills RAM at roughly
# $10 per GB-month and every megabyte resident is billed for every minute the
# container is awake.
#
# -slim rather than the full python image: same Python, no build toolchain,
# roughly 900MB smaller. We install no packages that need compiling.
FROM python:3.11-slim

WORKDIR /app

# Dependencies are copied and installed BEFORE the source. Docker caches each
# layer, so editing api.py rebuilds only the last two layers instead of
# reinstalling every package. Turns a 90-second rebuild into a 3-second one.
COPY requirements-serve.txt .

# --no-cache-dir stops pip keeping a second copy of every wheel inside the
# image, which would roughly double its size for no benefit.
RUN pip install --no-cache-dir -r requirements-serve.txt

# Note what is NOT installed: pandas, scikit-learn's full stack, shap, gensim.
# Those are training-time only. requirements-serve.txt is the boundary.
COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Railway assigns a port at runtime via $PORT and it is NOT known at build
# time, so the CMD must read it dynamically. Defaulting to 8000 keeps
# `docker run -p 8000:8000` working locally.
# Shell form (not exec form) so ${PORT} is expanded by the shell.
CMD uvicorn rci.api:app --host 0.0.0.0 --port ${PORT:-8000}
