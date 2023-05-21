import requests
import io
from PIL import Image


class Client:
    def __init__(self, server_url):
        '''
        Принимает url сервера для обращения.
        '''
        self.server_url = server_url

    def process_url(self, img_url, mode):
        '''
        Метод обработки изображений по ссылке.
        mode принимает значения 'blur' или 'crop'.

        Возвращает объект типа PIL.Image
        '''

        response_model = requests.post(
            f'{self.server_url}/process',
            json = {'url':img_url,
                    'mode':mode}
        )

        return Image.open(io.BytesIO(response_model.content))
