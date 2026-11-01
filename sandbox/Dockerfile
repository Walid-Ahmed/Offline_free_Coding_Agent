# Prepare once while online; the agent never builds or pulls images while testing.
FROM python:3.13-slim
RUN python -m pip install --no-cache-dir 'pytest>=8,<10'
ENV PYTHONDONTWRITEBYTECODE=1
