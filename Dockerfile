FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install with configurable extras (default: biology)
ARG EXTRAS="biology"
RUN pip install --no-cache-dir -e ".[$EXTRAS]"

# Create data directory for persistent downloads
RUN mkdir -p /root/.ct/data

VOLUME /root/.ct/data

ENTRYPOINT ["ct"]
