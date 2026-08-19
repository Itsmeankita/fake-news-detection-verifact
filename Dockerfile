# VeriFact Pro — container image
FROM python:3.11-slim

WORKDIR /app

# System deps needed by matplotlib/wordcloud (font rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev libpng-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data at build time (not at first request)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
