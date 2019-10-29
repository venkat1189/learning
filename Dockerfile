FROM python:3-slim

WORKDIR /delphi

RUN pip install flask

COPY . /delphi

CMD ["python", "run.py"]

