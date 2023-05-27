from locust import HttpUser, task
import random


urls = [
    r'https://00.img.avito.st/image/1/1.TS0C-La44cQ0USPBFPVsPjla48K8WWPMdFzjxrJR6c60.tJU3U0ZP3BlosloS0te97PuyC3k47oSVhQaJvtWq0io',
    r'https://60.img.avito.st/image/1/1.ZSRRcra4yc1nxUvAF1dNI2TQy8vv00vbZ97Lz-Hbwcfn.jfQln5nWSg4VETgb7owRGLiLgtSFdKu4KkN2vL6XKaw',
    r'https://50.img.avito.st/image/1/1.oaLQqLa4DUvmAc9Onrbyx-MKD01uCY9DpgwPSWABBUFm.3hFvAfDTU231YBeZyNfRQCkzkCPw3g3GCTr1PN2_jxg',
    r'https://70.img.avito.st/image/1/1.vMhO17a4ECF4ftIkOqGzjXt1EifwdpIpOHMSI_5-GCv4.YHhhIatQHjUmyfP6KDXaNYt5kW1DurOuVFvjOHMS9x4',
    r'https://60.img.avito.st/image/1/1.YRPe17a4zfrofg__kMJJOe51z_xgdk_yqHPP-G5-xfBo.U0htBgI-0-Hz7EmKeDR3LI-cKhXd1z-yYgl7AiBq3nA',
    r'https://30.img.avito.st/image/1/1.xtCY6ra5ajmuQ6g8mIX453tJbDMsyWL7KUloPSRDYDs.gQlB4xwXpezHKDwRdmvKgsAKCF-0OQdSMxjKliambng',
    r'https://30.img.avito.st/image/1/1.gINi2ba4LGpUcO5vbqvoklx7LmzceK5iFH0uaNJwJGDU.g0OntuAbjE1x_hIpMM4SvQ2MQ0-dCzljqgU7pAc3EOI',
    r'https://40.img.avito.st/image/1/1.6vJk77a4RhtSRoQePLyrhF1NRB3aTsQTEktEGdRGThHS.ZIOwpVMqmo9OFT5XkaXWUbBtTwVxcAuxdpKjmgwGqhg',
]

modes = ['blur', 'crop', 'both']


class PicturePostUser(HttpUser):

    def get_image(self):
        self.client.post('/process',
                         json={'url': random.choice(urls),
                               'mode': random.choice(modes)
                               }
                        )

    def health(self):
        self.client.get('/health')

    @task
    def stress(self):
        health_data_ratio = 0.01
        if random.uniform(0., 1.) < health_data_ratio:
            self.health()
        else:
            self.get_image()
