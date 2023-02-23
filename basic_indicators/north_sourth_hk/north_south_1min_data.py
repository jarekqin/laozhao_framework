import pandas as pd
import numpy as np

from datetime import timedelta, datetime
from time import time
import requests
import os

from omk.core.orm_db import EnginePointer
from basic_indicators.sql_toolkits.read_save_toolkits import save_to_sql
from basic_indicators.sql_toolkits.data_db import NORTHTOSOUTHINFLOW, NORTHTOSOUTHNETBUY, SOUTHTONORTHINFLOW, \
    SOUTHTONORTHNETBUY

con = EnginePointer.picker('finance_database')


def get_1min_north_and_south_in_flow_data(save_type='csv', save_path=None):
    cookies = {
        'cowCookie': 'true',
        'qgqp_b_id': '970c2cb78038fab9a65a353ca1e23aa8',
        'st_si': '15051098622212',
        'emshistory': '%5B%22%E4%B8%8A%E8%AF%8150ETF%E6%9C%9F%E6%9D%83%22%5D',
        'st_pvi': '24639492505499',
        'st_sp': '2022-04-17%2022%3A26%3A19',
        'st_inirUrl': 'https%3A%2F%2Fwww.baidu.com%2Flink',
        'st_sn': '7',
        'st_psi': '20220417230140406-113300300964-8795433551',
        'st_asi': 'delete',
        'intellpositionL': '1522.39px',
        'intellpositionT': '746px',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://data.eastmoney.com/',
        'Sec-Fetch-Dest': 'script',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Edg/100.0.1185.44',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Microsoft Edge";v="100"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    response = requests.get(
        'https://push2.eastmoney.com/api/qt/kamt.rtmin/get?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56&ut=b2884a393a59ad64002292a3e90d46a5&cb',
        headers=headers, cookies=cookies)

    temp = response.json()
    s2n = temp['data']['s2n']
    n2s = temp['data']['n2s']

    s2n_datetime = []
    s2n_sh_net_income = []
    s2n_sz_net_income = []
    s2n_sh_remianed_amt = []
    s2n_sz_remained_amt = []
    northen_amount = []
    for line in s2n:
        date, sh_net_income, sh_remained, sz_income, sz_remained, northen_amt = line.split(',')
        temp = [sh_net_income, sh_remained, sz_income, sz_remained, northen_amt]

        for i in range(len(temp)):
            if len(temp[i]) == 1:
                temp[i] = temp[i].replace('-', '0.0')
        sh_net_income, sh_remained, sz_income, sz_remained, northen_amt = temp

        s2n_datetime.append(pd.to_datetime(date))
        s2n_sh_net_income.append(np.float32(sh_net_income) / 1e4)
        s2n_sh_remianed_amt.append(np.float32(sh_remained) / 1e4)
        s2n_sz_net_income.append(np.float32(sz_income) / 1e4)
        s2n_sz_remained_amt.append(np.float32(sz_remained) / 1e4)
        northen_amount.append(np.float32(northen_amt) / 1e4)
    s2n_df = pd.DataFrame(
        {'datetime': s2n_datetime, 'sh_net_in_flow': s2n_sh_net_income, 'sh_remained': s2n_sh_remianed_amt,
         'sz_net_in_flow': s2n_sz_net_income, 'sz_remained': s2n_sz_remained_amt, 'northen_in_flow': northen_amount
         })
    s2n_df.set_index('datetime', inplace=True)

    n2s_datetime = []
    n2s_sh_net_income = []
    n2s_sz_net_income = []
    n2s_sh_remianed_amt = []
    n2s_sz_remained_amt = []
    southern_amount = []
    for line in n2s:
        date, sh_net_income, sh_remained, sz_income, sz_remained, northen_amt = line.split(',')
        temp = [sh_net_income, sh_remained, sz_income, sz_remained, northen_amt]

        for i in range(len(temp)):
            if len(temp[i]) == 1:
                temp[i] = temp[i].replace('-', '0.0')
        sh_net_income, sh_remained, sz_income, sz_remained, northen_amt = temp

        n2s_datetime.append(pd.to_datetime(date))
        n2s_sh_net_income.append(np.float32(sh_net_income) / 1e4)
        n2s_sh_remianed_amt.append(np.float32(sh_remained) / 1e4)
        n2s_sz_net_income.append(np.float32(sz_income) / 1e4)
        n2s_sz_remained_amt.append(np.float32(sz_remained) / 1e4)
        southern_amount.append(np.float32(northen_amt) / 1e4)
    n2s_df = pd.DataFrame(
        {'datetime': n2s_datetime, 'hk_sh_net_in_flow': n2s_sh_net_income, 'hk_sh_remained': n2s_sh_remianed_amt,
         'hk_sz_net_in_flow': n2s_sz_net_income, 'hk_sz_remained': n2s_sz_remained_amt,
         'southen_in_flow': southern_amount
         })
    n2s_df.set_index('datetime', inplace=True)

    if save_path is not None:
        if save_type == 'csv':
            n2s_df.to_csv(os.path.join(save_path, '%s_n2s.csv' % n2s_df.index.max().strftime('%Y-%m-%d')),
                          encoding='utf-8')
            s2n_df.to_csv(os.path.join(save_path, '%s_s2n.csv' % s2n_df.index.max().strftime('%Y-%m-%d')),
                          encoding='utf-8')
        elif save_type == 'excel':
            n2s_df.to_excel(os.path.join(save_path, '%s_n2s.xlsx' % n2s_df.index.max().strftime('%Y-%m-%d')),
                            encoding='utf-8')
            s2n_df.to_excel(os.path.join(save_path, '%s_s2n.xlsx' % s2n_df.index.max().strftime('%Y-%m-%d')),
                            encoding='utf-8')
        elif save_type == 'sql':
            save_to_sql(n2s_df, NORTHTOSOUTHINFLOW, con)
            save_to_sql(s2n_df, SOUTHTONORTHINFLOW, con)
        else:
            raise TypeError('save_type only supports "csv/excel/sql"')


def get_1min_north_and_south_net_buy(save_type='csv', save_path=None):
    cookies = {
        'cowCookie': 'true',
        'qgqp_b_id': '970c2cb78038fab9a65a353ca1e23aa8',
        'st_si': '15051098622212',
        'emshistory': '%5B%22%E4%B8%8A%E8%AF%8150ETF%E6%9C%9F%E6%9D%83%22%5D',
        'st_pvi': '24639492505499',
        'st_sp': '2022-04-17%2022%3A26%3A19',
        'st_inirUrl': 'https%3A%2F%2Fwww.baidu.com%2Flink',
        'st_sn': '7',
        'st_psi': '20220417230140406-113300300964-8795433551',
        'st_asi': 'delete',
        'intellpositionL': '1522.39px',
        'intellpositionT': '746px',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5',
        'Connection': 'keep-alive',
        # Requests sorts cookies= alphabetically
        # 'Cookie': 'cowCookie=true; qgqp_b_id=970c2cb78038fab9a65a353ca1e23aa8; st_si=15051098622212; emshistory=%5B%22%E4%B8%8A%E8%AF%8150ETF%E6%9C%9F%E6%9D%83%22%5D; st_pvi=24639492505499; st_sp=2022-04-17%2022%3A26%3A19; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=7; st_psi=20220417230140406-113300300964-8795433551; st_asi=delete; intellpositionL=1522.39px; intellpositionT=746px',
        'Referer': 'https://data.eastmoney.com/',
        'Sec-Fetch-Dest': 'script',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Edg/100.0.1185.44',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Microsoft Edge";v="100"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    response = requests.get(
        'https://push2.eastmoney.com/api/qt/kamtbs.rtmin/get?fields1=f1,f2,f3,f4&fields2=f51,f54,f52,f58,f53,f62,f56,f57,f60,f61&ut=b2884a393a59ad64002292a3e90d46a5&cb',
        headers=headers, cookies=cookies)
    temp = response.json()

    s2n = temp['data']['s2n']
    n2s = temp['data']['n2s']

    s2n_datetime = []
    s2n_sh_buy = []
    s2n_sz_buy = []
    s2n_sh_sell = []
    s2n_sz_sell = []
    north_buy = []
    north_sell = []
    sh_net = []
    sz_net = []
    north_net = []
    for line in s2n:
        date, sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value, \
        sz_buy_value, sz_sell_value, north_buy_value, north_sell_value = line.split(',')
        temp = [sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value,
                sz_buy_value, sz_sell_value, north_buy_value, north_sell_value]

        for i in range(len(temp)):
            if len(temp[i]) == 1:
                temp[i] = temp[i].replace('-', '0.0')

        sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value, \
        sz_buy_value, sz_sell_value, north_buy_value, north_sell_value = temp

        s2n_datetime.append(pd.to_datetime(date))
        s2n_sh_buy.append(np.float32(sh_buy_value) / 1e4)
        s2n_sz_buy.append(np.float32(sz_buy_value) / 1e4)
        s2n_sh_sell.append(np.float32(sh_sell_value) / 1e4)
        s2n_sz_sell.append(np.float32(sz_sell_value) / 1e4)
        north_buy.append(np.float32(north_buy_value) / 1e4)
        north_sell.append(np.float32(north_sell_value) / 1e4)
        sh_net.append(np.float32(sh_net_value) / 1e4)
        sz_net.append(np.float32(sz_net_value) / 1e4)
        north_net.append(np.float32(north_net_value) / 1e4)
    s2n_df = pd.DataFrame(
        {'datetime': s2n_datetime, 's2n_sh_buy': s2n_sh_buy, 's2n_sz_buy': s2n_sz_buy,
         's2n_sh_sell': s2n_sh_sell, 's2n_sz_sell': s2n_sz_sell, 'north_buy': north_buy,
         'north_sell': north_sell, 'sh_net': sh_net, 'sz_net': sz_net, 'north_net': north_net
         })
    s2n_df.set_index('datetime', inplace=True)

    n2s_datetime = []
    n2s_hk_sh_buy = []
    n2s_hk_sz_buy = []
    n2s_hk_sh_sell = []
    n2s_hk_sz_sell = []
    south_buy = []
    south_sell = []
    hk_sh_net = []
    hk_sz_net = []
    north_net = []
    for line in n2s:
        date, sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value, \
        sz_buy_value, sz_sell_value, north_buy_value, north_sell_value = line.split(',')
        temp = [sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value,
                sz_buy_value, sz_sell_value, north_buy_value, north_sell_value]

        for i in range(len(temp)):
            if len(temp[i]) == 1:
                temp[i] = temp[i].replace('-', '0.0')

        sh_net_value, sh_buy_value, sz_net_value, sh_sell_value, north_net_value, \
        sz_buy_value, sz_sell_value, north_buy_value, north_sell_value = temp

        n2s_datetime.append(pd.to_datetime(date))
        n2s_hk_sh_buy.append(np.float32(sh_buy_value) / 1e4)
        n2s_hk_sz_buy.append(np.float32(sz_buy_value) / 1e4)
        n2s_hk_sh_sell.append(np.float32(sh_sell_value) / 1e4)
        n2s_hk_sz_sell.append(np.float32(sz_sell_value) / 1e4)
        south_buy.append(np.float32(north_buy_value) / 1e4)
        south_sell.append(np.float32(north_sell_value) / 1e4)
        hk_sh_net.append(np.float32(sh_net_value) / 1e4)
        hk_sz_net.append(np.float32(sz_net_value) / 1e4)
        north_net.append(np.float32(north_net_value) / 1e4)
    n2s_df = pd.DataFrame(
        {'datetime': n2s_datetime, 'n2s_hk_sh_buy': n2s_hk_sh_buy, 'n2s_hk_sz_buy': n2s_hk_sz_buy,
         'n2s_hk_sh_sell': n2s_hk_sh_sell, 'n2s_hk_sz_sell': n2s_hk_sz_sell, 'south_buy': south_buy,
         'south_sell': south_sell, 'hk_sh_net': hk_sh_net, 'hk_sz_net': hk_sz_net, 'south_net': north_net
         })
    n2s_df.set_index('datetime', inplace=True)

    if save_path is not None:
        if save_type == 'csv':
            n2s_df.to_csv(os.path.join(save_path, '%s_n2s_net_buy.csv' % n2s_df.index.max().strftime('%Y-%m-%d')),
                          encoding='utf-8')
            s2n_df.to_csv(os.path.join(save_path, '%s_s2n_net_buy.csv' % s2n_df.index.max().strftime('%Y-%m-%d')),
                          encoding='utf-8')
        elif save_type == 'excel':
            n2s_df.to_excel(os.path.join(save_path, '%s_n2s_net_buy.xlsx' % n2s_df.index.max().strftime('%Y-%m-%d')),
                            encoding='utf-8')
            s2n_df.to_csv(os.path.join(save_path, '%s_s2n_net_buy.xlsx' % s2n_df.index.max().strftime('%Y-%m-%d')),
                          encoding='utf-8')
        elif save_type == 'sql':
            save_to_sql(n2s_df, NORTHTOSOUTHNETBUY, con)
            save_to_sql(s2n_df, SOUTHTONORTHNETBUY, con)
        else:
            raise TypeError('save_type only supports "csv/excel/sql"')


if __name__ == '__main__':
    import time
    from datetime import datetime

    # get_1min_north_and_south_in_flow_data('csv', 'E:\\老赵分析框架\\north_south_1min_data')
    # get_1min_north_and_south_net_buy('csv', 'E:\\老赵分析框架\\north_south_1min_data')
    # get_1min_north_and_south_in_flow_data('sql', 'E:\\老赵分析框架\\north_south_1min_data')
    # get_1min_north_and_south_net_buy('sql', 'E:\\老赵分析框架\\north_south_1min_data')

    while True:
        if datetime.now().hour >= 15 and datetime.now().minute >= 0 and datetime.now().second >= 0:
            get_1min_north_and_south_in_flow_data('sql', 'E:\\老赵分析框架\\north_south_1min_data')
            get_1min_north_and_south_net_buy('sql', 'E:\\老赵分析框架\\north_south_1min_data')
            break
        elif (datetime.now().strftime('%H:%M') >= '11:30') and (datetime.now().strftime('%H:%M') <= '13:00'):
            print('中午休盘中...')
            time.sleep(2)
        else:
            get_1min_north_and_south_in_flow_data('csv', 'E:\\老赵分析框架\\north_south_1min_data')
            get_1min_north_and_south_net_buy('csv', 'E:\\老赵分析框架\\north_south_1min_data')
            print(datetime.now())
            time.sleep(2)

