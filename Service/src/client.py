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
        response_img = requests.get(img_url)

        response_model = requests.post(
            f'{self.server_url}/process/mode={mode}',
            data=io.BytesIO(response_img.content)
        )

        return Image.open(io.BytesIO(response_model.content))

    def process_img(self, img_path, mode):
        '''
        Метод обработки изображений по локальному пути.
        mode принимает значения 'blur' или 'crop'.

        Возвращает объект типа PIL.Image
        '''
        img = Image.open(img_path)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        byte_im = buffer.getvalue()

        response_model = requests.post(
            f'{self.server_url}/process/mode={mode}',
            data=byte_im
        )

        return Image.open(io.BytesIO(response_model.content))
