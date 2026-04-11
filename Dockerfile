# Use a lightweight Python base image that matches the project requirement
FROM python:3.13-slim

# Set environment variables for Python
# PYTHONDONTWRITEBYTECODE=1 prevents Python from writing .pyc files
# PYTHONUNBUFFERED=1 ensures logs bypass the buffer and hit the cloud console immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install uv globally
RUN pip install --no-cache-dir uv

# Copy the dependency lockfile and pyproject.toml
COPY pyproject.toml uv.lock ./

# Install dependencies directly to the system python environment
# Use uv sync --system or export the lockfile to maintain pinned versions
RUN uv export --format requirements-txt > requirements.txt && \
    uv pip install --system -r requirements.txt && \
    rm requirements.txt

# Copy the remaining project files
COPY . .

# Set the default command to execute the main analysis/auto loop
CMD ["python", "app.py", "auto"]
