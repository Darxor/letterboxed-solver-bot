ARG TAG=3.12-alpine

FROM python:${TAG} AS builder

ARG POETRY_VERSION=1.7.1

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apk add --no-cache \
        curl \
        gcc \
        g++ \
        gfortran \
        libressl-dev \
        musl-dev \
        libffi-dev \
        openblas-dev \
        cmake && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile=minimal && \
    source $HOME/.cargo/env && \
    pip install --no-cache-dir poetry==${POETRY_VERSION} 

WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-root --no-ansi --without tests

# ---------------------------------------------------------------------

FROM python:${TAG}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8800
WORKDIR /app

RUN apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    openblas-dev

# copy the venv folder from builder image 
COPY --from=builder /app/.venv ./.venv
COPY . .

CMD ["/bin/ash", "-c", "python main.py"]
