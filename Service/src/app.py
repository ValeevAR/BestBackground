from flask import Flask, request, send_file, after_this_request
import logging
from PIL import Image
import io
from service import init_models, get_jewellery_image


# Инициализация приложения и моделей
app = Flask(__name__)
model_detection, model_mask = init_models()


@app.route('/process/mode=<string:mode>', methods=['POST'])
def process(mode):
    '''
    Метод для обработки Byte-файлов.
    mode принимает 2 значения: "crop" и "blur"

    Возвращает файл в формате Byte-массива.

    Пример обращения:
    requests.post(f'<server_url>/process/mode=<mode>',
                  data = <io.BytesIO file>
                )
    '''
    req_data = request.get_data()
    img = Image.open(io.BytesIO(req_data))

    buffer = io.BytesIO()

    if mode not in ['blur', 'crop']:
        raise ValueError('Mode has to be either "crop" or "blur"')

    res = get_jewellery_image(img, model_detection, model_mask)

    if mode == 'crop':
        res[0]['cropped_image'].save(buffer, 'PNG')
    elif mode == 'blur':
        res[0]['image_segmented'].save(buffer, 'PNG')
    buffer.seek(0)
    # file_path = file_path.getvalue()

    return send_file(buffer, mimetype='image/png')

@app.route('/health')
def health():
    '''
    Проверка состояния активности сервера.
    '''
    return {'result':True}


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
