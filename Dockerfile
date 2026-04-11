FROM python:3.11-slim

WORKDIR /app

# System deps for TA-Lib and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy dependency spec first for layer caching
COPY pyproject.toml .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# Copy app source
COPY . .

# Create non-root user
RUN addgroup --system autotrader && adduser --system --ingroup autotrader autotrader
RUN chown -R autotrader:autotrader /app
USER autotrader

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "null"]
