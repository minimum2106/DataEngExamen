FROM python:3.8-alpine
WORKDIR /app

COPY experiments /experiments/
# WORKDIR .

COPY main.py .
COPY README.md .

RUN pip install -r requirements.txt
CMD python main.py%