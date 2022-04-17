import requests
from lxml import etree
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Query, sessionmaker
from sqlalchemy import and_, or_
import matplotlib.pyplot as plt
import dataframe_image as dfi
from pandas import MultiIndex

from omk.core.orm_db.jarvis import CommodityPx100PPI
from omk.core.orm_db import EnginePointer
from omk.toolkit.job_tool import JobManager
from omk.events import Event
from omk.interface import AbstractJob
from omk.utils.const import ProcessDocs, ProcessType, EVENT, TIME, JobType
from jarvis.utils import FOLDER, my_candle, mkdir
from omk.core.vendor.RQData import RQData

import matplotlib as mpl
import seaborn as sns
import rqdatac as rq

from warnings import filterwarnings

from omk.toolkit.calendar_ import get_trading_dates

filterwarnings('ignore')


def idom(url):
    u'''
    功能:
      获取url对应的Dom对象, 返回Element Tree对象
    param:
      url: 要访问的网页的url地址
    return:
      Element Tree对象, 网页的DOM节点树
    '''
    # url='http://www.100ppi.com/monitor/'
    resp = requests.get(url)  # 发送 GET 请求, 返回response对象
    html = resp.text  # 响应对象的文本属性
    # encoding = resp.encoding
    dom = etree.HTML(html)  # 返回root node of the document
    return dom


def crawle_firstClassCommodity(url,
                               dom,
                               anchor_path='//div[starts-with(@class,"title_b2")]/span/a'):
    u'''
    功能:
      从生意社网站, 抓取一级分类的大宗商品名称, 商品价格监测的连接地址
    param:
      dom: \n
      anchor_path:
    return:
      dataframe of 第一级分类商品的数据框, 列名称为:['id','firstClassCommodity','url_link']
    '''

    # F12=> 审查元素=> 右击元素的属性区=>选择edit Attribute=> select and copy
    #   class="title_b2 w666 fl", 对于这种复杂的属性值, 匹配时需要用starts-with函数
    #   比如: [starts-with(@class, "title_b2")], 否则属性无法匹配
    anchors = dom.xpath(anchor_path)
    # xpath方法返回的总是list对象, 即便是只有一个元素
    # 用锚点标签(计算过程里的中间变量或者叫上下文节点)的xpath方法, 分别获取它的属性值和上层的文本.
    # 获取它的href属性值: .xpath('@href')[0]
    # 获取它的上层文本: .xpath('text()')[0]
    firstClassCommodity = []
    link = []
    for each in anchors:  # each means each_anchor, 每一个锚点标签
        # print
        # print '字典的项目{}'.format(each.attrib.items())
        # print '字典的键',each.attrib.keys()
        # print '字典的值', each.attrib.values()
        # print '该a标签的href属性的值=', each.attrib.get('href')
        # print each.xpath('text()')[0]
        # .xpath(//text()) .xpath(/text()) .xpath(text()), .xpath(text())[0]
        # 注意上述的四个个表达式得到截然不同的结果, 最后一个的才是正确的
        firstClassCommodity.append(each.xpath('text()')[0])
        # link.append(url + each.attrib.get('href') ) # 获取属性, 通过操作元素字典的方法
        # print each.xpath('@href')[0] # 获取属性, 通过元素的xpath方法
        link.append(url + each.xpath('@href')[0])

    data = {'firstClassCommodity': firstClassCommodity,
            'url_link': link,
            'id': [int(e[-2:]) for e in link]
            }
    df = pd.DataFrame(data,
                      # index= data['id'],
                      columns=['id', 'firstClassCommodity', 'url_link']
                      )
    # df=df.sort_index()
    df = df.sort_values(by='id')  # 按id列递增排序
    df = pd.DataFrame(df.values, columns=df.columns)
    s = 'NY|YS|GT|HG|XS|FZ|JC|NF'
    df['prefix'] = s.split('|')
    # print df
    return df


def crawle_secondClassCommodity(dom, table_path='//table[@width="99%"]'):
    table = dom.xpath(table_path)[0]
    # 获取所有table下的行
    rows = table.xpath('tr')
    type_name = []
    dates = []
    data_dict = {}
    counter = 1
    for row in rows:
        print('Checking NO.%d row!' % counter)
        # anchor = row.xpath('//td[@class="w1"]/a/@href')
        # 抓取表格head上的时间
        temp_dates = row.xpath('//td[@width="12%"]/text()')
        if len(temp_dates) > 0 and len(dates) == 0:
            for i in range(len(temp_dates)):
                month = temp_dates[i][:2]
                day = temp_dates[i][3:5]
                year = datetime.today().year
                dates.append(datetime(year, int(month), int(day)).date())

        # 判断是否包含粗字体,长度等于1
        if len(row.xpath('td/a/b/text()')) == 1 and row.xpath('td/a/b/text()')[0] not in data_dict:
            temp_type_name = row.xpath('td/a/b/text()')[0]
            if temp_type_name not in data_dict:
                data_dict[temp_type_name] = pd.DataFrame()
            print('NO.%d rows checked successfully and start type: %s !' % (counter, temp_type_name))
            counter += 1
            continue
        if counter == 132:
            print()
        # 判断是否为合约数据
        if len(row.xpath('td/a/text()')) == 1:
            for i, j in zip(range(len(dates)), range(2, 5)):
                if row.xpath('td')[j].text is not None and '-' != row.xpath('td')[j].text:
                    temp_df = pd.DataFrame(
                        {'name': row.xpath('td/a/text()')[0],
                         'type_': temp_type_name,
                         'px': np.float(row.xpath('td')[j].text)},
                        index=[dates[i]],
                    )
                    data_dict[temp_type_name] = pd.concat([data_dict[temp_type_name], temp_df], axis=0)
                else:
                    continue

        print('NO.%d rows checked successfully!' % counter)
        counter += 1

    return data_dict


class UpdateCommodityPX100(AbstractJob):
    def __init__(self, picker_name='finance_database'):
        if picker_name is None:
            self.engine = EnginePointer.get_engine()
        else:
            self.engine = EnginePointer.picker(picker_name)

    def register_event(self, event_bus, job_uuid, debug=True):
        if RQData.check_for_trading_date():
            event_bus.add_listener(Event(
                event_type=EVENT.AM0700,
                alert=True,
                func=self.main,
                p_type=ProcessType.Jarvis,
                des='Commodity Infom Update',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

            event_bus.add_listener(Event(
                event_type=EVENT.AM0900,
                alert=True,
                func=self.main,
                p_type=ProcessType.Jarvis,
                des='Commodity Infom Update',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

            event_bus.add_listener(Event(
                event_type=EVENT.PM1200,
                alert=True,
                func=self.main,
                p_type=ProcessType.Jarvis,
                des='Commodity Infom Update',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

            event_bus.add_listener(Event(
                event_type=EVENT.PM0300,
                alert=True,
                func=self.main,
                p_type=ProcessType.Jarvis,
                des='Commodity Infom Update',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

            event_bus.add_listener(Event(
                event_type=EVENT.PM1100,
                alert=True,
                func=self.main,
                p_type=ProcessType.Jarvis,
                des='Commodity Infom Update',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

        if datetime.today().isoweekday() == 5:
            event_bus.add_listener(Event(
                event_type=EVENT.PM0410,
                alert=True,
                func=self.cut_data,
                p_type=ProcessType.Jarvis,
                des='Spots VS Futures Plot',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
                cut_end_date=None,
                save_path='E:\\commidity_monitor_plot'
            ))

    def main(self):
        # 获取当天最新数据结构
        url = 'http://www.100ppi.com/monitor'
        dom = idom(url)
        data = crawle_secondClassCommodity(dom)

        # 删除过去2天的历史数据
        session = sessionmaker(self.engine)()
        pre_three_days = pd.period_range(datetime.today() - timedelta(2), datetime.today())
        pre_three_days = [x.to_timestamp().date() for x in pre_three_days]
        session.query(CommodityPx100PPI).filter(CommodityPx100PPI.PxDate.in_(pre_three_days)).delete('fetch')
        session.commit()
        session.close()

        for contract_name in data:
            if data[contract_name].shape[0] > 0:
                data[contract_name].reset_index().rename(columns={'index': 'px_date'}).to_sql(
                    CommodityPx100PPI.table_name(), con=self.engine,
                    schema=CommodityPx100PPI.schema(),
                    if_exists='append', index=False
                )

        self.plot()

        # 从本地读取 Excel
        # temp_table = pd.read_csv(os.path.join(FOLDER.Syn_output,'商品价格.csv'), encoding='gbk')
        # temp_table.columns=['name','type_','px_date','px']
        # temp_table.px_date=[pd.to_datetime(x).date() for x in temp_table.px_date]
        # temp_table.dropna().to_sql(
        #     CommodityPx100PPI.table_name(), con=self.engine,
        #     schema=CommodityPx100PPI.schema(),
        #     if_exists='append', index=False
        # )

    def plot(self, contract_names=None, save_path='E:\\commidity_monitor_plot'):

        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        sns.set(font_scale=1.5, font='SimHei')

        if contract_names is None:
            contract_names = [
                '炼焦煤', '焦炭', '动力煤', '液化天然气',
                '聚合MDI', 'PX', '甲醇', '乙二醇', '钛白粉',
                '不锈钢板', '螺纹钢', '铁矿石(澳)', '铁矿石(印)', '铁矿石(巴)',
                '水泥', '阔叶木浆',
                '大豆', '豆粕', '油菜籽', '白糖', '生猪', '玉米',
                '多晶硅', '金属硅', '玻璃', '磷矿石', '钴', '镍', '锰硅',
                '氢氧化锂', '磷酸铁锂', '金属钕'
            ]

            mkdir(os.path.join(save_path, 'futures_test', datetime.today().strftime('%Y-%m-%d')))

            # 读取指定的合约数据
            data = pd.read_sql(Query(CommodityPx100PPI).filter(CommodityPx100PPI.Name.in_(contract_names)).statement,
                               con=self.engine).set_index('px_date').replace(0, np.nan)
            data = data.fillna(method='ffill')
            for n in data.name.unique():
                if n == '锰硅':
                    print(n)
                temp_data = data[data.name == n].sort_index()
                type_ = temp_data.type_.unique()[0]
                temp_data = temp_data['px'].to_frame()
                temp_data.rename(columns={'px': 'close'}, inplace=True)
                temp_data['open'] = temp_data['close'].shift(1)
                temp_data['high'] = np.maximum(temp_data.close, temp_data.open)
                temp_data['low'] = np.minimum(temp_data.close, temp_data.open)
                temp_data['pct'] = (temp_data.close / temp_data.close.iloc[0] - 1) * 100
                fig = my_candle3(temp_data, None, '%s_%s_现货价格K线图' % (temp_data.index[-1].strftime('%Y-%m-%d'), n),
                                 '时间', '价格')

                # temp_data['px'].plot(figsize=(15, 12), rot=45)
                # plt.ylabel('收盘价', fontdict={'size': '14'})
                # plt.xlabel('日期', fontdict={'size': '14'})
                # plt.title('时间: %s-%s \n 行业: %s-%s' % (
                #     temp_data.index[0].strftime('%Y-%m-%d'),
                #     temp_data.index[-1].strftime('%Y-%m-%d'),
                #     temp_data.type_.unique()[0], n
                # ), fontdict={'size': '14'})
                if save_path is None:
                    fig.show()
                    plt.close()
                else:
                    fig.savefig(os.path.join(save_path, 'futures_test', temp_data.index[-1].strftime('%Y-%m-%d'),
                                             '%s_%s.png' % (type_, n)))
                    plt.close()

    def cut_data(self, contract_list=None, cut_end_date=None,
                 save_path='E:\\commidity_monitor_plot'):
        if cut_end_date is None:
            cut_end_date = datetime.today()
        cut_start_date = cut_end_date - timedelta(120)

        if contract_list is None:
            contract_list = {
                '光伏': ['多晶硅', '金属硅', '玻璃'],
                '锂电': ['磷矿石', '钴', '镍', '锰硅', '氢氧化锂', '磷酸铁锂'],
                '稀土': ['金属钕'],
                '能源': ['炼焦煤', '焦炭', '动力煤', '液化石油气'],
                '农副': ['豆一', '豆粕', '油菜籽', '白糖', '生猪', '玉米'],
                '化工': ['甲醇', '乙二醇', '聚合MDI', 'PX', '钛白粉']
            }

            futures_mapping_list1 = ['玻璃', '镍', '焦炭', '动力煤', '液化石油气', '豆一', '豆粕', '油菜籽',
                                     '白糖', '生猪', '玉米', '甲醇', '乙二醇']

        # 先处理现货
        temp_list = [y for x in contract_list for y in contract_list[x] if y not in futures_mapping_list1]
        data = pd.read_sql(Query(CommodityPx100PPI).filter(and_(
            CommodityPx100PPI.PxDate >= cut_start_date.date(),
            CommodityPx100PPI.PxDate <= cut_end_date.date()
        )).filter(CommodityPx100PPI.Name.in_(temp_list)).statement, con=self.engine).set_index('px_date').drop(
            columns='key')

        return_df = pd.DataFrame()
        for industry in contract_list:
            temp_data = data[data.name.isin(contract_list[industry]) == True]
            for contract in contract_list[industry]:
                if contract in futures_mapping_list1:
                    continue
                print(contract)
                temp_data2 = temp_data[temp_data.name == contract]
                try:
                    temp_dict = {
                        'name': contract,
                        'style': industry,
                        'lastest_price': temp_data2.loc[cut_end_date.date(), 'px'],
                        'chg_cur_1week': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                cut_end_date.date() - timedelta(7), 'px']) /
                            temp_data2.loc[cut_end_date.date() - timedelta(7), 'px'] * 100, 2)) + '%',
                        'chg_cur_2weeks': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                cut_end_date.date() - timedelta(14), 'px']) /
                            temp_data2.loc[cut_end_date.date() - timedelta(14), 'px'] * 100, 2)) + '%',
                        'chg_cur_4weeks': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                cut_end_date.date() - timedelta(28), 'px']) /
                            temp_data2.loc[cut_end_date.date() - timedelta(28), 'px'] * 100, 2)) + '%',
                        'last_price_change_date': temp_data2.px.diff().replace(0, np.nan).dropna().index[-1].strftime(
                            '%Y-%m-%d'),
                    }
                except Exception as e:
                    temp_dict = {
                        'name': contract,
                        'style': industry,
                        'lastest_price': temp_data2.loc[cut_end_date.date(), 'px'],
                        'chg_cur_1week': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                get_trading_dates(cut_end_date - timedelta(7), cut_end_date)[
                                    0], 'px']) /
                            temp_data2.loc[get_trading_dates(cut_end_date - timedelta(7), cut_end_date)[
                                    0], 'px'] * 100, 2)) + '%',
                        'chg_cur_2weeks': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                get_trading_dates(cut_end_date - timedelta(14), cut_end_date)[
                                    0], 'px']) /
                            temp_data2.loc[get_trading_dates(cut_end_date - timedelta(14), cut_end_date)[
                                    0], 'px'] * 100, 2)) + '%',
                        'chg_cur_4weeks': str(round(
                            (temp_data2.loc[cut_end_date.date(), 'px'] - temp_data2.loc[
                                get_trading_dates(cut_end_date - timedelta(28), cut_end_date)[
                                    0], 'px']) /
                            temp_data2.loc[get_trading_dates(cut_end_date - timedelta(28), cut_end_date)[
                                    0], 'px'] * 100, 2)) + '%',
                        'last_price_change_date': temp_data2.px.diff().replace(0, np.nan).dropna().index[-1].strftime(
                            '%Y-%m-%d'),
                    }
                return_df = pd.concat([return_df, pd.DataFrame(temp_dict, index=[cut_end_date.date()])],
                                      axis=0).dropna()

        # 后处理期货
        # 获取中英文映射
        whole_futures = rq.all_instruments(type='Future', market='cn')
        futures_needed = whole_futures[
            whole_futures.symbol.str.contains('|'.join(futures_mapping_list1)) == True].set_index('symbol')

        futures_needed.index = [x.replace(x[-4:], '') for x in futures_needed.index]
        futures_mapping_list2 = futures_needed['underlying_symbol'].to_dict()
        futures_mapping_list2 = {x: y for x, y in futures_mapping_list2.items() if x in futures_mapping_list1}

        if len(futures_mapping_list2) != len(futures_mapping_list1):
            raise ValueError('futures mapping length are not same!')

        for k, v in futures_mapping_list2.items():
            temp_dominant = rq.get_dominant_future(v, start_date=cut_start_date.date(),
                                                   end_date=cut_end_date.date()).unique()
            whole_dominant_data = rq.get_price(temp_dominant, start_date=cut_start_date, end_date=cut_end_date,
                                               expect_df=True).close.reset_index()
            temp_data = whole_dominant_data[whole_dominant_data.order_book_id == temp_dominant[0]].set_index('date')
            for contract in temp_dominant[1:]:
                temp_data2 = whole_dominant_data[whole_dominant_data.order_book_id == contract].set_index('date')
                temp_data = pd.concat([temp_data, temp_data2.loc[temp_data2.index.difference(temp_data.index)]], axis=0)
            temp_dict = {
                'name': k + '*',
                'style': [ind for ind in contract_list if k in contract_list[ind]][0],
                'lastest_price': temp_data.loc[cut_end_date.date(), 'close'],
                'chg_cur_1week': str(round(
                    (temp_data.loc[cut_end_date.date(), 'close'] - temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(7), cut_end_date)[0], 'close']) /
                    temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(7), cut_end_date)[0], 'close'] * 100,
                    2)) + '%',
                'chg_cur_2weeks': str(round(
                    (temp_data.loc[cut_end_date.date(), 'close'] - temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(14), cut_end_date)[0], 'close']) /
                    temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(14), cut_end_date)[0], 'close'] * 100,
                    2)) + '%',
                'chg_cur_4weeks': str(round(
                    (temp_data.loc[cut_end_date.date(), 'close'] - temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(28), cut_end_date)[0], 'close']) /
                    temp_data.loc[
                        rq.get_trading_dates(cut_end_date.date() - timedelta(28), cut_end_date)[0], 'close'] * 100,
                    2)) + '%',
                'last_price_change_date': '-',
            }
            return_df = pd.concat([return_df, pd.DataFrame(temp_dict, index=[temp_data2.index.max().date()])],
                                  axis=0).dropna()
        # 输出图样
        return_df = return_df.reset_index().rename(columns={'index': 'date'}).set_index(['style', 'name']).sort_index()
        max_date = return_df.date.max()
        return_df.drop(columns='date', inplace=True)
        return_df.index.names = [None, None]
        return_df.columns = MultiIndex.from_product(
            [['%s_期现货价格面板' % max_date.strftime('%Y-%m-%d')], return_df.columns.to_list()])

        mkdir(os.path.join(save_path, '%s' % max_date.strftime('%Y-%m-%d')))
        dfi.export(return_df,
                   os.path.join(save_path, '%s' % max_date.strftime('%Y-%m-%d'), 'spots_futures_plot.jpg'),
                   max_rows=return_df.shape[0]
                   )


# %%
if __name__ == '__main__':
    # from jarvis.jobs.commidity_monitor import UpdateCommodityPX100
    # JobManager.install_job('Commodity Infom Update', UpdateCommodityPX100, JobType.Module, activate=True)
    RQData.init()
    model = UpdateCommodityPX100()
    model.main()
    # model.plot()
    # model.cut_data()

    # manager = JobManager()
    # model.register_event(event_bus=manager.event_bus, job_uuid=None, debug=True)
    # if datetime.now().hour == 7 and datetime.now().minute <= 59 and datetime.now().second <= 59:
    #     manager.event_bus.event_queue_reload(EVENT.AM0700)
    #     manager.event_bus.sequential_publish()
    # elif datetime.now().hour == 9 and datetime.now().minute <= 59 and datetime.now().second <= 59:
    #     manager.event_bus.event_queue_reload(EVENT.AM0900)
    #     manager.event_bus.sequential_publish()
    # elif datetime.now().hour == 12 and datetime.now().minute <= 59 and datetime.now().second <= 59:
    #     manager.event_bus.event_queue_reload(EVENT.AM1200)
    #     manager.event_bus.sequential_publish()
    # elif datetime.now().hour == 15 and datetime.now().minute <= 59 and datetime.now().second <= 59:
    #     manager.event_bus.event_queue_reload(EVENT.PM0300)
    #     manager.event_bus.sequential_publish()
    #     manager.event_bus.event_queue_reload(EVENT.PM0410)
    #     manager.event_bus.sequential_publish()
    # elif datetime.now().hour == 23 and datetime.now().minute <= 59 and datetime.now().second <= 59:
    #     manager.event_bus.event_queue_reload(EVENT.PM1100)
    #     manager.event_bus.sequential_publish()
    # else:
    #     pass
