FROM python:3.12.6-slim

WORKDIR /code/app

COPY . /code/app/

RUN pip install -r /code/app/req.txt

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]