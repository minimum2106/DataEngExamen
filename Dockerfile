FROM python:3.8-alpine
WORKDIR /app

COPY experiments /experiments/
COPY experiment_original_data /experiment_original_data/
# WORKDIR .

COPY main.py .
COPY README.md .
COPY embeddings.pickle .
COPY labels.pickle .
COPY utils.py .

RUN pip install -r requirements.txt
CMD python main.py