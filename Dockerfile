FROM python:3.11.8-slim
WORKDIR /app

# Disable Poetry virtualenv — install everything in system Python
ENV POETRY_VIRTUALENVS_CREATE=false
ENV HOST=0.0.0.0
ENV PORT=8000

# Install torch CPU-only via pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install poetry
RUN pip install poetry

# Dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --compile

# Copy project
COPY . .
RUN poetry install --no-root --compile

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0"]