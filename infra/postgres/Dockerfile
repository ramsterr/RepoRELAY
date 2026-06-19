FROM pgvector/pgvector:pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        flex \
        bison \
        git \
        postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*
