# Stage 1: Install dependencies
FROM python:3.9-slim as builder

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies with network configuration
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set work directory
WORKDIR /app

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]