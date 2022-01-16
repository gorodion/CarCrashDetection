from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions
import time

# button = driver.find_element('xpath', '//button[@class="btn-save-later"]')
# try:
#     button.click()
# except exceptions.WebDriverException:
#     time.sleep(5)
#     button.click()

CITIES = ['', 'kg', 'mg', 'st', 'ks', 'ptk', 'suo', 'ldh', 'sg', 'ndv', 'mrsk', 'sd']


out = open('parsed_urls.txt', 'w')
driver = webdriver.Safari()
for city in CITIES:
    print(city, file=out)
    driver.get(f'https://moidom.citylink.pro/publist/{city}')

    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Архив"))
        )
    except exceptions.TimeoutException:
        print('link on "Archive" not found')
        continue
    vid_urls = [el.get_attribute('href') for el in driver.find_elements_by_link_text('Архив')]
    if len(vid_urls) != 12:
        print('Total', len(vid_urls), f'places for {city} found')
    for vid_url in vid_urls:
        #TODO random intervals
        driver.get(vid_url)
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "camera_html5_api"))
            )
        except exceptions.TimeoutException:
            print(f'camera on {vid_url} not found')
            continue
        stream = driver.find_element_by_id('camera_html5_api')
        src_url = stream.find_element_by_xpath('source').get_attribute('src')
        print(src_url, file=out)
out.close()
# driver.get('https://moidom.citylink.pro/pub/103/archive/202201151344-202201151345')
# TODO если нет такого видео

# vid_urls = [el.get_attribute('href') for el in driver.find_elements('xpath', '//ul[@class="items-grid"]/li/a')]
# vid_url = vid_url + '/202201151419-202201151426'

# cookies = driver.get_cookies()
# for cookie in cookies:
#     driver.add_cookie(cookie)
