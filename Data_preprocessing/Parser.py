import random
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image

BASIC_PATH = './raw_data6/'


def getdata(url: str) -> str:
    r = requests.get(url)
    return r.text


def get_links_from_url(url: str) -> dict:
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')

    links = {}
    for item in soup.find_all('img'):
        print()
        print(item)

        try:
            link = item['src']

            if link[0] == '/':
                link = 'https://kira-scrap.ru' + link

            if link[:4] != 'http':
                link = 'https://' + link

            print(link)
            # if 'files' in link:
            name = str(random.randint(0, 999999999)) + '.png'
            links[name] = link
        except:
            pass

    return links


def save_img_from_links(links: dict, basic_path: str):
    for name, link in links.items():

        try:
            image0 = requests.get(link)
            img = Image.open(BytesIO(image0.content))
            if img.mode == 'RGBA':
                img_file = open(basic_path + name, 'wb')
                img_file.write(image0.content)
                img_file.close()
                print('+', end='')
            print(' no RGBA ', end='')

        except:
            print('-', end='')
    print()


if __name__ == '__main__':
    total_img_number = 0

    for n in range(1, 10):  # 10
        print(f'page number {n}')
        # site = f'https://www.pngmart.com/ru/image/tag/jewellery/page/{n}'
        # site = f'https://www.pngarts.com/ru/explore/category/accessories/jewellery/page/{n}'
        # site = f'https://imgpng.ru/img/jewelry/ring'
        # site = f'https://imgpng.ru/img/jewelry/necklace'
        # site = f'https://imgpng.ru/img/jewelry/jewelry'
        # site = f'https://imgpng.ru/img/jewelry/pearl'
        site = f'https://kira-scrap.ru/dir/svadba/dragocennosti_krome_kolec/508-{n}'

        links = get_links_from_url(site)

        print(links)

        img_number = len(links)
        print(f'downloading {img_number} images ')
        save_img_from_links(links, BASIC_PATH)
        total_img_number += img_number
        print('downloaded')
        print()

    print(f'downloaded {total_img_number} images')
