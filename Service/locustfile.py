import random

from locust import HttpUser, task
from src.client import EXAMPLE_URLS


MODES = ['blur', 'crop', 'both']
HEALTH_RATIO = 0.01


class PicturePostUser(HttpUser):

    def get_image(self):
        '''
        Функция для отправки запроса по получению изображений
        '''
        self.client.post('/process',
                         json={'url': random.choice(EXAMPLE_URLS),
                               'mode': random.choice(MODES)
                               }
                         )

    def health(self):
        '''
        Редкий запрос для проверки состояния сервера.
        '''
        self.client.get('/health')

    @task
    def stress(self):
        '''
        Функция нагрузочного тестирования. С вероятностью
        HEALTH_RATIO отправляет проверку здоровья, все остальное
        время присылаются случайные запросы на обработку изображений.
        '''
        if random.uniform(0., 1.) < HEALTH_RATIO:
            self.health()
        else:
            self.get_image()
