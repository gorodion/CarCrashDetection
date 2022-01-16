from lxml.html import fromstring as to_tree
import pandas as pd

data = pd.DataFrame([])
tree = to_tree(open('citylink2.html', encoding='utf-8').read())
elements = tree.xpath('//div[@id="items"]//ytd-grid-video-renderer//a[@id="video-title"]')
data['title'] = [el.text for el in elements]
data['url'] = [el.get('href') for el in elements]
data.to_csv('urls.csv', index=False)
print(data)
