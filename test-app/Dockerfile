FROM python:3.7

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt


CMD ["gunicorn" , "-b" , "0.0.0.0:9000" , "wsgi:app"]

