import os
import random

from locust import HttpUser, task


MODES = ['blur', 'crop', 'both']
GET_IMAGE_RATIO = 100
GOOD_STATUS = [2]
ERROR_STATUS = [4, 5]

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'src', 'example_urls.txt'), 'r') as file:
    EXAMPLE_URLS = file.read().split('\n')


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
