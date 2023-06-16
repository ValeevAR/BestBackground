FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /bestbg_app

COPY . ./bestbg_app

RUN apt-get update && apt-get -y install curl && pip install -r ./bestbg_app/requirements.txt

EXPOSE 8080

CMD [ "python3", "./bestbg_app/src/app.py" ]