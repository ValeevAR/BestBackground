FROM python:3.9.7

WORKDIR /bestbg_app

COPY . ./bestbg_app

RUN pip install -r ./bestbg_app/requirements.txt

EXPOSE 8080

CMD [ "python3", "./bestbg_app/src/app.py" ]