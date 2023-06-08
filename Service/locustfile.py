import random

from locust import HttpUser, task
from src.client import EXAMPLE_URLS


MODES = ['blur', 'crop', 'both']
GET_IMAGE_RATIO = 100
GOOD_STATUS = [2]
ERROR_STATUS = [4, 5]


class PicturePostUser(HttpUser):

    @task(GET_IMAGE_RATIO)
    def get_image(self):
        '''
        Функция для отправки запроса по получению изображений
        '''
        with self.client.post('/process',
                              json={'url': random.choice(EXAMPLE_URLS),
                                    'mode': random.choice(MODES)
                                    },
                              catch_response=True
                              ) as response:
            if response.status_code // 100 in GOOD_STATUS:
                response.success()
            elif response.status_code // 100 in ERROR_STATUS:
                response.failure(str(response.content))

    @task(1)
    def health(self):
        '''
        Редкий запрос для проверки состояния сервера.
        '''
        self.client.get('/health')
