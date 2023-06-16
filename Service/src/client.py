import io
import os
from base64 import decodebytes

import requests
from PIL import Image

GOOD_STATUS = [2]
ERROR_STATUS = [4, 5]

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'example_urls.txt'), 'r') as file:
    EXAMPLE_URLS = file.read().split('\n')


def decode_img(decoded_img):
    '''
    Обработка изображения в PIL.Image из текста, переданного
    в запросе.

    :param decoded_img: изображение закодированное в base64.
    :return: PIL.Image с декодированным изображением.
    '''
    return Image.open(io.BytesIO(decodebytes(decoded_img.encode())))


class Client:
    def __init__(self, server_url, raise_errors=True):
        '''
        Принимает url сервера для обращения.
        raise_errors = False обрабатывает ошибки как обычные
        результаты, True -- вызывает ошибку.

        :param server_url: строчка с адресом сервера
        :param raise_errors: если ответом сервера является ошибка и
            raise_errors == False: обрабатывает ошибки как обычные
            результаты, raise_errors == True -- вызывает ошибку в коде
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

        :param img_url: строчка с URL изображения в интернете.
        :param mode: режим обработки, подаваемый на сервер.
            Может быть "crop", "blur" или "both".
        :return: массив присланных сервером изображений или ответ
            сервера с текстом ошибки и статусом ошибки.
        '''

        response = requests.post(
            f'{self.server_url}/process',
            json={'url': img_url,
                  'mode': mode
                  }
        )
        if response.status_code//100 in GOOD_STATUS:
            return [
                decode_img(decoded_img)
                for decoded_img in response.json()['result']
            ]
        elif response.status_code//100 in ERROR_STATUS:
            if self.raise_errors:
                msg = (f'{response.status_code} {(response.content).decode()}')
                raise RuntimeError(msg)
            return response.content, response.status_code


# Набор ссылок для нагрузочного тестирования и презентации модели.
EXAMPLE_URLS = [
    r'https://00.img.avito.st/image/1/1.TS0C-La44cQ0USPBFPVsPjla48K8WWPMdFzjxrJR6c60.tJU3U0ZP3BlosloS0te97PuyC3k47oSVhQaJvtWq0io',
    r'https://60.img.avito.st/image/1/1.ZSRRcra4yc1nxUvAF1dNI2TQy8vv00vbZ97Lz-Hbwcfn.jfQln5nWSg4VETgb7owRGLiLgtSFdKu4KkN2vL6XKaw',
    r'https://50.img.avito.st/image/1/1.oaLQqLa4DUvmAc9Onrbyx-MKD01uCY9DpgwPSWABBUFm.3hFvAfDTU231YBeZyNfRQCkzkCPw3g3GCTr1PN2_jxg',
    r'https://70.img.avito.st/image/1/1.vMhO17a4ECF4ftIkOqGzjXt1EifwdpIpOHMSI_5-GCv4.YHhhIatQHjUmyfP6KDXaNYt5kW1DurOuVFvjOHMS9x4',
    r'https://50.img.avito.st/image/1/1.RL3cTra56FTq5ypR2E95sgXt7l5obeCWbe3qUGDn4lY.PeHaM6vTjgQSdVwR51Dxjg9PDXbqHlC_8xbRWUNB-Fg',
    r'https://60.img.avito.st/image/1/1.YRPe17a4zfrofg__kMJJOe51z_xgdk_yqHPP-G5-xfBo.U0htBgI-0-Hz7EmKeDR3LI-cKhXd1z-yYgl7AiBq3nA',
    r'https://30.img.avito.st/image/1/1.xtCY6ra5ajmuQ6g8mIX453tJbDMsyWL7KUloPSRDYDs.gQlB4xwXpezHKDwRdmvKgsAKCF-0OQdSMxjKliambng',
    r'https://30.img.avito.st/image/1/1.gINi2ba4LGpUcO5vbqvoklx7LmzceK5iFH0uaNJwJGDU.g0OntuAbjE1x_hIpMM4SvQ2MQ0-dCzljqgU7pAc3EOI',
    r'https://40.img.avito.st/image/1/1.6vJk77a4RhtSRoQePLyrhF1NRB3aTsQTEktEGdRGThHS.ZIOwpVMqmo9OFT5XkaXWUbBtTwVxcAuxdpKjmgwGqhg',
    r'https://20.img.avito.st/image/1/1.PYXNgLa4kWz7KVNpu8Qh_eoik2pzIRNkuySTbn0pmWZ7.ZoBwtiXUVdxM0vkxWosX4OAHg81iER3FB_sZbNPrhsQ',
    r'https://40.img.avito.st/image/1/1.42lnGra4T4BRs42FLXmwDFS4TYbZu82IEb5NgtezR4rR.Ouw_Nnvoxb9oAWGFZIKxKGyfTcA6-kD9e8bEwoI3ePo',
    r'https://20.img.avito.st/image/1/1.3b8XALa4cVYhqbNTIX-Z2C-ic1CpofNeYaRzVKepeVyh.GcjR1D6drHrH8TfB8oS1v8gyHjy0FZpCgF0jBTlijpk',
    r'https://90.img.avito.st/image/1/1.vpcmMLa4En4QmdB7VhbCuh6SEHiYkZB2UJQQfJaZGnSQ.C4Gm-TOeJ82S9s5ucsSPFblfihEAQCSpVmb05saWbcA',
    r'https://00.img.avito.st/image/1/1.OyQue7a5l80Y0lXIVDc6Z_zYkceaWJ8Pn9iVyZLSnc8.iN0qIZNZmBTGHDAChDK_6HXj-Rdv0GuZT-oAMmkZEBU',
    r'https://30.img.avito.st/image/1/1.RBHhQra46PjX6yr95W9gatng6v5f42rwl-bq-lHr4PJX.AdTfxpoXVuReCxyFrr-6Bsw5FMGKrJu328P5HSNjt6A',
    r'https://80.img.avito.st/image/1/1.L2-UZba4g4aizEGDxnVYf7LHgYAqxAGO4sGBhCTMi4wi.pUBaneIbNu43disU1NrAeZgDyU7MhNoGov8EzzC0zWY',
    r'https://90.img.avito.st/image/1/1.dpIgMra52nsWmxh-QCMlr-WR3HGUEdK5kZHYf5yb0Hk.trGg-jKKJTmagQu369Y3IWYpnkKlfuRRffquouxXR30',
    r'https://90.img.avito.st/image/1/1.aLfNu7a4xF77EgZbkd4AhOsZxlhzGkZWux_GXH0SzFR7.Y0lMDDUFgkFt4NOa00-hgsQHB-Yz-ZiJIZo_H40FALg',
    r'https://60.img.avito.st/image/1/1.WWcX2ba49Y4hcDeLKYR0EiN794ipeHeGYX33jKdw_YSh.NARs-YJ6jLL-K0jix_7SvL0T2fpo_JAkHq5ebOFRpIU',
    r'https://50.img.avito.st/image/1/1.NXX1gra4mZzDK1uZk_ZMEtMgm5pLIxuUgyabnkUrkZZD.phKe4CzrxxllC08Bh_p6-13-AYygSVSWhHT8cmp-Dz0',
    r'https://40.img.avito.st/image/1/1.vRUcira4EfwqI9P5aqDzZSwoE_qiK5P0ai4T_qwjGfaq.VmvPjgL81IXf7VnmVykCKx7XaYdKWK2iOPAuzsRtZaw',
    r'https://20.img.avito.st/image/1/1.9e2OLra4WQS4h5sB6HmfuLyMWwIwj9sM-IpbBj6HUQ44.VCpQIEEdcvllWxv0MUUHLhdC61W3e2BJSLf7rzjQ1as',
    r'https://90.img.avito.st/image/1/1.oU78cba5DafK2M-isg_sMCTSC61IUgVlTdIPo0DYB6U.dIsz84TR5VXEK-v1MO-xEcNHEoH2ezmvWInaNgOxamc',
    r'https://80.img.avito.st/image/1/1.iMVFa7a5JCxzwuYpWVHqk7bIIibxSCzu9MgmKPnCLi4.BAfYepRzTE9DvG655Q_6PzNtsUAqVtkICIpqj1pWmIc',
    r'https://00.img.avito.st/image/1/1.uQ-43La5FeaOddfjxva5Kmt_E-wM_x0kCX8X4gR1H-Q.9YTdwR8nuOafg9GkNVo911q8kDnOv1q6T1gIwC80HNE',
    r'https://20.img.avito.st/image/1/1.YQGT_La5zeilVQ_t3fl4DlBfy-In38UqIl_P7C9Vx-o.yyZJ820guJUpfo4wnaZssTUtThW7Rx4QeWJruJtLx5g',
    r'https://30.img.avito.st/image/1/1.Azmrmra4r9CdM23VoepjdJg4rdYVOy3Y3T6t0hszp9od.heC5evW4tTYaUNUuGd_rh46gc1fyYNtPJIAgC6W8BJM'
    ]
