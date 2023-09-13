from mimetypes import init

from WindPy import w
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 全日成交额
# w.wsd("881001.WI", "amt", "2023-07-14", "2023-08-12", "unit=1")
def Wind_A_Index_30min(start_date, end_date, code='8841388.WI', Any='amt', save_path=none):
    if init:
        w.start()
    start_date, end_date = pd.to_datetime(start_date).strftime('%Y-%m-%d'), pd.to_datetime(end_date).strftime(
        '%Y-%m-%d')
    data = w.wsi(f"{code}", f"{Any}", f"{start_date} ", f"{end_date} ", "BarSize=30")
    data = pd.DataFrame(np.transpose(data.Data), index=np.transpose(data.Times), columns=data.Codes)
    data = pd.DataFrame()
    data.reset_index().to_feather(save_path)
    return data


# 日成交额/A股总市值
def company_size(start_date, end_date, save_path):
    # A股总市值
    # w.edb("M0331246", "2022-08-13", "2023-08-13","Fill=Previous")
    if init:
        w.start()
    start_date, end_date = pd.to_datetime(start_date).strftime('%Y-%m-%d'), pd.to_datetime(end_date).strftime(
        '%Y-%m-%d')
    # 全A市值
    data = w.wsd("881001.WI", "mkt_cap_ard", f"{start_date}", f"{end_date}",, "unit=1;Currency=CNY")
    data1 = pd.DataFrame(np.transpose(data.Data), index=data.Times, columns=data.Codes)
    # 全A平均等权指数日成交额
    dataa = w.wsd("881001.WI", "amt", f"{start_date}", f"{end_date}", "unit=1")
    data2 = pd.DataFrame(np.transpose(dataa.Data), index=dataa.Times, columns=dataa.Codes)
    data_all = data1.Data / data2.Data
    data_all = pd.DataFrame()
    data_all.to_csv(save_path)
    # print(data_all)

    return data_all


def plot_(data, xlabel, ylabel, title, fontsize=18, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    data.plot(color='r', ax=ax, label='ratio_test!!!')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(loc='best', fontsize=fontsize)
    # ax.xaxis.set_major_locator(local) # 自动左右对齐
    plt.xticks(rotation=60)  # x轴上的字符串，按顺时针转动60度
    plt.title(title, fontsize=14)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


if __name__ == "__main__":
    # print(single_col('000001.SH','2023-08-01','2023-08-07','close',True))
    # print(multi_col('000001.SH', '2023-08-01', '2023-08-07', ['close', 'open'], True))
    # print(Wind_A_Index_30min('2023-07-01', '2023-08-07',code='8841388.WI', Any='amt'))
    # print(company_size('2023-07-01', '2023-08-13', True))
    data = pd.read_csv(r'E:\kaifa_test\company_size.csv')
    data.set_index('time', inplace=True)  # 把时间当作index
    data.index = [x.strftime('%Y-%m-%d %H:%M:%S') for x in pd.to_datetime(data.index)]  # 日期转换
    print(data)
    plot_(data, 'Time', 'Value', 'Company size / trading amounts')
