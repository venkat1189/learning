FROM python:3-slim

COPY . /delphi/

WORKDIR /delphi

CMD ["python", "test.py"]

