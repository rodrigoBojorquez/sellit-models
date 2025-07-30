FROM python:3.11.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv
ENV PIPENV_VENV_IN_PROJECT=false
RUN pipenv install --deploy --ignore-pipfile

COPY . .

EXPOSE 8000
CMD ["pipenv", "run", "fastapi", "run", "api/main.py", "--host", "0.0.0.0", "--port", "8000"]
