FROM python:slim-bookworm

# Install dependencies
RUN apt-get update
RUN apt-get install -y \
    ffmpeg \
    git

WORKDIR /app
COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY src src
COPY config config
RUN ls -lsa
RUN pwd

CMD ["python3", "src/main.py"]
