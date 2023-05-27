import requests
import io
from PIL import Image
from base64 import decodebytes


def decode_img(decoded_img):
    '''
    Обработка изображения в PIL.Image из текста, переданного
    в запросе.
    '''
    return Image.open(io.BytesIO(decodebytes(decoded_img.encode())))


class Client:
    def __init__(self, server_url, raise_errors=True):
        '''
        Принимает url сервера для обращения.
        handle_errors = False обрабатывает ошибки как обычные
        результаты, True -- вызывает ошибку.
        '''
        self.server_url = server_url
        self.raise_errors = raise_errors

    def process_url(self, img_url, mode):
        '''
        Метод обработки изображений по ссылке.
        mode принимает значения 'blur', 'crop' или 'both'.

        Возвращает массив объектов типа PIL.Image.
        В случае ошибки возвращает ее как обычный ответ сервиса
        или вызывает ошибки на выбор пользователя (см. __init__).
        '''

        response_model = requests.post(
            f'{self.server_url}/process',
            json = {'url':img_url,
                    'mode':mode}
        )
        if response_model.status_code//100 == 2:
            return [
                decode_img(decoded_img)
                for decoded_img in response_model.json()['result']
            ]
        elif response_model.status_code//100 in [4, 5]:
            if self.raise_errors:
                err_msg = f'{response_model.status_code} {(response_model.content).decode()}'
                raise RuntimeError(err_msg)
            return response_model.content, response_model.status_code