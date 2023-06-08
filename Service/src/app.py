import io
import logging
from base64 import encodebytes

import requests
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from service import get_jewellery_image, init_models


MODES = ['blur', 'crop', 'both']


# Инициализация приложения и моделей
app = Flask(__name__)
model_detection, model_mask = init_models(use_gpu=True)


def decode_img(img):
    '''
    Декодер для PIL-изображений в строчный формат.

    :param img: PIL.Image изображение для инкодинга
    :return: строка с изображением, закодированным в b64
    '''
    buffer = io.BytesIO()
    img.save(buffer, 'PNG')
    encoded_img = encodebytes(buffer.getvalue()).decode('ascii')
    return encoded_img


def read_img(url):
    '''
    Загрузка изображения по ссылке в виде PIL.Image.

    :param url: URL на изображение
    :return: PIL.Image с изображением.
    '''
    req_img = requests.get(url).content
    img = Image.open(io.BytesIO(req_img))
    return img


@app.route('/process', methods=['POST'])
def process():
    '''
    Метод для обработки запросов.
    Принимает json с ключевыми слвоами "url" и  "mode".
    "url" должен быть ссылкой на изображение в доступе у сервера.
    mode принимает 3 значения: "crop" и "blur" или "both".

    Возвращает файлы в json по аргументу result.
    Изображения кодируются в виде строк в b64 для передачи в json
    в виде массива изображений.

    Пример обращения:
    requests.post(f'<server_url>/process/',
                  json = {'url': <LINK_TO_IMG>, 'mode': <MODE>}
                )
    '''
    req_json = request.json
    req_url = req_json.get('url')
    mode = req_json.get('mode')
    try:
        img = read_img(req_url)
    except UnidentifiedImageError:
        return ("Error! Could not extract image from the url", 400)

    if mode not in MODES:
        return (f'Error! Mode has to be in {MODES}.', 400)

    try:
        res = get_jewellery_image(img, model_detection, model_mask)
    except RuntimeError:
        return ("Error! Server could not process the image", 500)

    imges = []
    try:
        if mode == 'both':
            imges.append(decode_img(res[0]['cropped_image']))
            imges.append(decode_img(res[0]['image_segmented']))
        elif mode == 'crop':
            imges.append(decode_img(res[0]['cropped_image']))
        elif mode == 'blur':
            imges.append(decode_img(res[0]['image_segmented']))
    except KeyError:
        return ("Error! Model didn't find an object on the image.", 500)

    return jsonify({'result': imges})


@app.route('/health')
def health():
    '''
    Проверка состояния активности сервера.
    '''
    return {'result': True}


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
