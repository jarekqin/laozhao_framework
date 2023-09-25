from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver

import os
import requests
import time


import pandas as pd
import numpy as np

CHROM_DRIVER=r'/media/qin/项目代码存放处/laozhao_framework/futures_info_scrapy/chromedriver'


def spot_daily_price(url, save_path=r'./'):
    # 初始化网页驱动
    bro = webdriver.Chrome()
    bro.get(r'https://user.smm.cn/login?referer=https://www.smm.cn/')
    time.sleep(5)

    usernameinput=bro.find_element('id','userName')
    usernameinput.send_keys('13002110402')
    time.sleep(3)
    passwrodinput = bro.find_element('id','password')
    time.sleep(3)
    passwrodinput.send_keys('65029289a')
    click_button=bro.find_element('id','user_account_password_login_button')
    click_button.click()


    today_ = datetime.today().strftime('%Y%m%d')
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    html_ = BeautifulSoup(resp.text, 'html.parser')
    columns_list = ['名称', '价格范围', '均价', '涨跌', '单位', '日期']
    # 获取对应信息第一次获取
    spot_info = html_.findAll('tr', {'class': ''})
    # 第二次获取
    spot_info2 = html_.findAll('tr', {'class': "line-bottom"})
    # 第三次获取
    spot_info3 = html_.findAll('tr', {'class': "line-top"})
    results = []
    # 按照品种遍历
    for info_ in spot_info + spot_info2 + spot_info3:
        # 先获取现货名字
        try:
            temp_spot_name = info_.find('a', {'class': 'href_label_module_spot 2asd'}).contents[0]
        except Exception as e:
            continue
        # 获取价格范围
        temp_spot_price_scale = '-'.join([info_.get('data-low'), info_.get('data-high')])
        # 获取均价
        temp_spot_mean_price = float(info_.get('data-average'))
        # 获取涨跌幅
        temp_spot_pct = None
        # 获取日期
        temp_date = pd.to_datetime(info_.get('data-date')).strftime('%Y-%m-%d')
        # 获取单位
        temp_unit = info_.find('td', {'class': 'c5'}).contents[0]
        results.append(
            [temp_spot_name, temp_spot_price_scale, temp_spot_mean_price, temp_spot_pct, temp_unit, temp_date])
    df = pd.DataFrame(results)
    df.columns = columns_list
    df = df.drop_duplicates(subset='名称')
    df.to_csv(os.path.join(save_path, today_ + '_spot_price_info.csv'),index=False)


if __name__ == '__main__':
    url = r'https://www.smm.cn/'
    spot_price_info = spot_daily_price(url)
