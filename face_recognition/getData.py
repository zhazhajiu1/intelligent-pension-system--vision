import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def download_images(query, save_path, num_images):
    # 设置Edge驱动
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # 使用EdgeDriver
    driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)

    url = f'https://image.baidu.com/search/index?tn=baiduimage&word={query}'
    driver.get(url)

    # 滚动页面以加载更多图片
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(10):  # 调整滚动次数以加载更多图片
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)  # 等待页面加载

    img_elements = driver.find_elements(By.CSS_SELECTOR, 'img.main_img')
    os.makedirs(save_path, exist_ok=True)
    count = 0
    downloaded_count = 0  # 用于跟踪已下载图片的数量
    for img_element in img_elements:
        if count >= num_images:
            break
        try:
            img_url = img_element.get_attribute('src')
            if img_url is None:
                img_url = img_element.get_attribute('data-src')
            if img_url is None:
                continue
            img_data = requests.get(img_url).content

            # 检查文件是否存在并递增count
            while os.path.exists(os.path.join(save_path, f'lyf_{count}.jpg')):
                count += 1

            with open(os.path.join(save_path, f'lyf_{count}.jpg'), 'wb') as f:
                f.write(img_data)
                print(f'成功获取{count}')

            downloaded_count += 1
            count += 1  # 递增count以准备下一张图片的文件名

        except Exception as e:
            print(f'无法获取图片: {e}')

    driver.quit()


query = '刘亦菲'
save_path = './data_test/volunteer/lyf'
num_images = 20
download_images(query, save_path, num_images)
