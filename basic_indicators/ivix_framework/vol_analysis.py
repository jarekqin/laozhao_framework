import os
import click
import bisect
from collections import defaultdict
from itertools import product
from datetime import datetime, timedelta, time
from omk.toolkit.calendar_ import get_trading_dates

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scipy import stats
from sqlalchemy import func
from sqlalchemy.orm import Query, sessionmaker

from omk.interface import AbstractJob, AbstractJarvisJob
from omk.core.orm_db import Base, truncate_table, EnginePointer
from omk.core.orm_db.jarvis import Volatility, IVix, OptionBasis
from omk.core.vendor.RQData import RQData
from omk.core.vendor.Wind import Wind
from omk.events import Event
from omk.toolkit.option_cal import Calculator
from omk.toolkit.mail import MailTool, MailType
from omk.utils.result import folder_init
from omk.utils.argument import CodeRegulator
from omk.utils.const import CODE, FolderName, InstrumentType, InstrumentAttr, JarvisFields, \
    OmakaFields, ProcessDocs, ProcessType, EVENT, TIME, OptionType, JobType

from jarvis.utils import concat_img, FOLDER
from omk.toolkit.job_tool import JobManager

plt.style.use('bmh')
mpl.rcParams['font.sans-serif'].insert(0, 'SimHei')
mpl.rcParams['font.sans-serif'].insert(0, 'Microsoft YaHei')
mpl.rcParams['axes.unicode_minus'] = False


class VolType:
    IVix = 'ivix'
    STD_Vol = 'std_vol'
    GK_Vol = 'gk_vol'
    PK_Vol = 'pk_vol'


class VolAnalysis(AbstractJob, AbstractJarvisJob):

    def __init__(self):

        self._eng = EnginePointer.picker('finance_database')
        self._code_list = [
            CODE.SH50ETF.rq, CODE.SSE50Index.rq,
            CODE.SH300ETF.rq, CODE.HS300Index.rq,
            CODE.ZZ500Index.rq
        ]
        self._window_list = [20, 40, 60]
        self._ivix_ud = [CODE.SH50ETF.rq, CODE.SH300ETF.rq]
        self._atm_multipliers = [10000.0, ]

        self._job_folder = None
        self._option_info = None
        self._option_quote = None
        self._option_day_bar_quote = None
        self._trading_dates = None
        self._ud_quote = None
        self._option_strike_lib = defaultdict(list)
        self._option_ttm_lib = defaultdict(list)

    def _set_job_folder(self, debug=False):
        self._job_folder = folder_init(os.path.join(
            FOLDER.Debug if debug else FOLDER.Prod, FolderName.Vol_Analysis, f'{TIME.live_today():%Y%m%d}'
        ))

    def register_event(self, event_bus, job_uuid, debug=False):

        if RQData.check_for_trading_date() or debug:
            event_bus.add_listener(Event(
                event_type=EVENT.PM0400 if not debug else EVENT.DEBUG,
                func=self.run,
                gap=None,
                alert=True,
                p_type=ProcessType.Jarvis,
                des=ProcessDocs.Jarvis_Vol_Analysis,
                job_uuid=job_uuid,
                ivix_freq='1m',
                ivix_bkwd_year=1,
                hist_bkwd_year=5,
                risk_free=0.03,
                debug=debug,
                retry_n=5,
                retry_freq='15m',
                callback=self._callback
            ))

    @staticmethod
    def _callback(event_bus, event):
        event.event_type = EVENT.PM1100
        event.callback = None
        event_bus.add_listener(event)

    @staticmethod
    def mkdir(file_path):
        folder = os.path.exists(file_path)
        if not folder:
            try:
                os.mkdir(file_path, mode=777)
            except Exception:
                return False
        else:
            return False
        return True

    def run(self, ivix_freq='1d', ivix_obs_days=30, ivix_bkwd_year=1, hist_bkwd_year=5, risk_free=0.03, verbose=False,
            debug=False):

        self._job_folder=os.path.join('E:\\vol%s' % TIME.today().strftime(
            '%Y%m%d'))
        VolAnalysis.mkdir(self._job_folder)

        self.update(ivix_freq, risk_free, verbose)

        merged_list = []
        un_merged_img_list = []

        attach_img_list = []

        for ud_code in self._ivix_ud:
            img_ivix, ivix_df = self.plot_ivix(ud_code, ivix_obs_days, ivix_bkwd_year, ivix_freq)
            img_vol_bias = self.plot_vol_basis(ud_code, ivix_df, ivix_bkwd_year)
            img_hist_vol, hist_vol = self.hist_vol_analysis(ud_code, hist_bkwd_year)
            hist_vol = pd.merge(
                hist_vol,
                ivix_df.resample('d').last().dropna().to_frame(VolType.IVix),
                how='outer', right_index=True, left_index=True
            )
            hist_vol.index = [x.date() for x in hist_vol.index]
            img_heat_map = self.plot_heatmap(ud_code, hist_vol)
            merged_list += self.merge_images(img_ivix + img_vol_bias, 'iVIX', ud_code.strip('.'))
            un_merged_img_list += img_hist_vol + img_heat_map

            attach_img_list += img_ivix + img_heat_map

        # # 沪深300指数
        # img_300_list, df_300 = self.hist_vol_analysis(CODE.HS300Index.rq, windows_list, bkwd_year)
        # img_300_list = img_300_list + self.plot_heatmap(CODE.HS300Index.rq, df_300)

        # 中证500指数
        img_500_hist_vol, hist_vol_500 = self.hist_vol_analysis(CODE.ZZ500Index.rq, hist_bkwd_year)
        img_500_heat_map = self.plot_heatmap(CODE.ZZ500Index.rq, hist_vol_500)

        attach_img_list += img_500_heat_map
        un_merged_img_list += img_500_hist_vol + img_500_heat_map

        merged_list += self.merge_images(un_merged_img_list, 'heatmap')
        merged_list += self.merge_images(un_merged_img_list, 'gk-std')
        merged_list += self.merge_images(un_merged_img_list, 'std_vol_his')
        merged_list += self.merge_images(un_merged_img_list, 'his_vol_20')
        merged_list += self.merge_images(un_merged_img_list, 'his_vol_40')
        merged_list += self.merge_images(un_merged_img_list, 'his_vol_60')

        # mail_tool = MailTool()
        # mail_tool.log_mail(
        #     ProcessType.Jarvis,
        #     MailType.Quant_Dev if debug else MailType.Inv_Dept,
        #     f'{TIME.live_today():%Y-%m-%d}: 波动率分析',
        #     image_addr=merged_list,
        #     attachment_addr=attach_img_list
        # )
        # print('Volatility analysis mail registered.')

    def plot_ivix(self, ud_code, ivix_obs_days=30, bkwd_year=1, freq='1d'):
        start_date = (pd.to_datetime(TIME.live_today()) - pd.DateOffset(years=bkwd_year)).date()
        ivix_obs_start = pd.to_datetime(TIME.live_today() - timedelta(days=ivix_obs_days))
        ivix_ser = self.load(ud_code, start_date, TIME.live_today(), VolType.IVix).set_index(
            OmakaFields.As_of_TS)[JarvisFields.iVIX.Eng].rename(VolType.IVix).sort_index()

        ivix_obs_start = min(filter(lambda x: x >= ivix_obs_start, ivix_ser.index))
        ivix_obs_ser = ivix_ser.loc[ivix_obs_start:]

        ud_df = RQData.get_history_bar(ud_code, ivix_obs_start, TIME.live_today(), freq=freq).loc[
                :, ivix_obs_ser.index, ud_code]

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig, ax = plt.subplots(figsize=(16, 8))

        h1 = ax.plot(ivix_obs_ser.values, color='b')
        ax2 = ax.twinx()
        ax2.grid(False)
        h2 = ax2.plot(ud_df['close'].values, color='r')
        ax.legend([h1[0], h2[0]], ['iVIX', f'{ud_code}'])

        n = ivix_obs_ser.shape[0]
        xtick_space = 240
        x_axis = ivix_obs_ser.index
        xtick = list(range(0, n, xtick_space))
        xtick_labels = [x_axis[i].date() for i in xtick]
        ax.set_xticks(xtick, minor=False)  # 设计刻度
        ax.set_xticklabels(xtick_labels, minor=False, rotation=30)
        xtick.insert(0, 0)
        xtick_labels.insert(0, str(x_axis[0]))

        ax.grid(True, ls='-.')
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

        ax.set_title(
            f"【iVIX_{ud_code}】中国波指: {ivix_obs_ser.index[0]:%Y-%m-%d}至{ivix_obs_ser.index[-1]:%Y-%m-%d}\n"
            f"最新iVIX: {ivix_ser.iloc[-1]:.2f}"
        )
        ax.set_xlabel("时间")
        ax.set_ylabel("iVIX")
        fig.autofmt_xdate()

        ivix_fig = os.path.join(self._job_folder, f"{ud_code.strip('.')}_iVIX.png")
        fig.savefig(ivix_fig, bbox_inches='tight')

        # 画ivix的直方图
        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(111)
        fig3, ax3 = plt.subplots(figsize=(16, 8))
        # vol_series = ivix_ser[VolType.IVix]
        ax3.hist(ivix_ser, range=(ivix_ser.min(), ivix_ser.max()), bins=20)
        ax3.grid(True)
        ax3.set_title(f"【iVIX_{ud_code}】中国波指直方分布图:{start_date:%Y-%m-%d}至{ivix_ser.index[-1]:%Y-%m-%d}\n")
        ax3.set_xlabel("iVIX")
        ax3.set_ylabel("频数")

        # 画出最新的标准差波动率所在的位置
        ax3.axvline(ivix_ser.iloc[-1], color='red')

        last_value = round(ivix_ser.iloc[-1], 2)
        percentile = round(stats.percentileofscore(ivix_ser, last_value), 2)

        # 添加注释：当前iVIX和当前iVIX所处的分位数
        ax3.annotate(
            "[{}]\niVIX={}\n分位数={}%".format(ivix_ser.index[-1], last_value, percentile),
            xy=(last_value, 25),
            xycoords='data', xytext=(15, 30),
            textcoords='offset points', arrowprops=dict(arrowstyle='->')
        )

        hist_fig = os.path.join(self._job_folder, f"{ud_code.strip('.')}_iVIX_his.png")
        fig3.savefig(hist_fig, bbox_inches='tight')
        return [ivix_fig, hist_fig], ivix_ser

    def plot_vol_basis(self, ud_code, ivix_df, bk_year=1, window=20):
        # sec_id = CODE.SH50ETF.rq
        # window = 20
        start_date = (ivix_df.index[0] - pd.Timedelta(days=window * 2)).date()
        _, concat_df = self.plot_his_vol(
            ud_code,
            RQData.get_history_bar(ud_code, start_date, TIME.live_today()).loc[:, :, ud_code],
            window,
            bk_year
        )
        concat_df = pd.merge(concat_df, ivix_df, how='outer', right_index=True, left_index=True)
        concat_df.index = [x.date() for x in concat_df.index]
        concat_df['vol_basis'] = concat_df[VolType.IVix] - concat_df[VolType.STD_Vol]
        end_date = concat_df.index[-1]
        n = concat_df.shape[0]

        def format_date(x, pos=None):
            ind = np.clip(int(x + 0.5), 0, n - 1)
            return concat_df.index[ind]

        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111)
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        # ax2.set_ylim([-30, 10])
        h1, = ax2.plot(range(n), concat_df['vol_basis'])

        ax22 = ax2.twinx()
        # ax22.set_ylim([0, 50])
        h2, = ax22.plot(range(n), concat_df[VolType.IVix], "--", linewidth=0.5, color='green')
        h3, = ax22.plot(range(n), concat_df[VolType.STD_Vol], "--", linewidth=0.5, color='grey')
        # 设置y轴范围

        ax2.legend([h1, h2, h3], ['diff', f'{VolType.IVix}', f'{VolType.STD_Vol}_{window}'])
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax2.set_title("【{}】_波动率基差(iVIX-std_{})(单位:%):截止{:%Y-%m-%d}".format(ud_code, window, end_date))
        ax2.set_xlabel("时间")
        ax2.set_ylabel("波动率差(%)")
        ax2.axhline(0, color='red')
        fig2.autofmt_xdate()
        fig_file = os.path.join(self._job_folder, f"{ud_code}_iVIX-std{window}.png")
        fig2.savefig(fig_file, bbox_inches='tight')
        return [fig_file]

    def plot_his_vol(self, sec_id, input_df, window, bk_year=1):
        input_df = input_df.copy()
        start, end = input_df.index.min(), input_df.index.max()

        vol_df = self.load(sec_id, start.date(), end.date(), window=window).pivot(
            OmakaFields.As_of_TS,
            JarvisFields.VolType.Eng,
            JarvisFields.VolValue.Eng
        ).reindex(columns=[VolType.STD_Vol, VolType.PK_Vol, VolType.GK_Vol]).sort_index()
        vol_df.index = [x.date() for x in vol_df.index]

        years_ago = datetime.now().date() - timedelta(days=365 * bk_year)
        vol_df = vol_df[vol_df.index > years_ago]

        last_update = vol_df.index[-1]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        n = len(vol_df)

        def format_date(x, pos=None):
            ind = np.clip(int(x + 0.5), 0, n - 1)
            return vol_df.index[ind]

        # 设置y轴范围，便于对比
        if window <= 20:
            ax.set_ylim([5, 45])
        elif window <= 40:
            ax.set_ylim([5, 35])
        else:
            ax.set_ylim([7, 32])
        handler_list = []
        for i, column in enumerate(list(vol_df.columns.values)):
            exec("h{}, = ax.plot(range(n), vol_df[column])".format(i))
            exec("handler_list+=[h{}]".format(i))
            pass
        ax2 = ax.twinx()
        handler, = ax2.plot(range(n), input_df['close'].iloc[-n:], "--", color='gold', linewidth=0.5)
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

        ax.legend(handler_list + [handler], list(vol_df.columns.values) + [sec_id])
        ax.set_title("【{}】_历史波动率_移动窗口{}（单位:交易日）:截止{}".format(sec_id, window, last_update))
        ax.set_xlabel("时间")
        ax.set_ylabel("波动率(%)")
        fig.autofmt_xdate()

        filepath = os.path.join(self._job_folder, f"{sec_id}_his_vol_{window}.png")
        fig.savefig(filepath, bbox_inches='tight')
        plt.close("all")
        # 如果window是20的话，增加gk_vol和std_vol的差的图
        if window == 20:
            path_list = self.plot_his_and_diff(sec_id, vol_df, window)
            return [filepath] + path_list, vol_df
        else:
            return [filepath], vol_df

    def plot_his_and_diff(self, sec_id, input_df, window=20):
        input_df = input_df.copy()

        n = len(input_df)
        start_date = input_df.dropna().index.values[0]
        end_date = input_df.index.values[-1]

        def format_date(x, pos=None):
            ind = np.clip(int(x + 0.5), 0, n - 1)
            return input_df.index.values[ind]

        # 画gk(20)和std(20)的差
        fig2 = plt.figure()

        ax2 = fig2.add_subplot(111)
        ax2.set_ylim([-30, 10])
        h1, = ax2.plot(range(n), input_df[VolType.GK_Vol] - input_df[VolType.STD_Vol])

        ax22 = ax2.twinx()
        ax22.set_ylim([0, 50])
        h2, = ax22.plot(range(n), input_df[VolType.GK_Vol], "--", linewidth=0.5, color='green')
        h3, = ax22.plot(range(n), input_df[VolType.STD_Vol], "--", linewidth=0.5, color='grey')

        ax2.axhline(0, color='red')
        # 设置y轴范围

        ax2.legend([h1, h2, h3], ['diff', VolType.GK_Vol, VolType.STD_Vol])
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax2.set_title("【{}】_GK波动率(20)-标准差波动率(20)(单位:%):截止{}".format(sec_id, end_date))
        ax2.set_xlabel("时间")
        ax2.set_ylabel("波动率差(%)")
        fig2.autofmt_xdate()
        filepath2 = os.path.join(self._job_folder, f"{sec_id}_gk-std_{window}.png")
        fig2.savefig(filepath2, bbox_inches='tight')

        # 画std(20)的直方图
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        vol_series = input_df[VolType.STD_Vol]
        ax3.hist(vol_series, range=(vol_series.min(), vol_series.max()))
        ax3.grid(True)
        ax3.set_title("【{}】_标准差波动率(20) 直方分布图:{}至{}".format(sec_id, start_date, end_date))
        ax3.set_xlabel("标准差波动率(%)")
        ax3.set_ylabel("频数")

        # 画出最新的标准差波动率所在的位置
        ax3.axvline(input_df[VolType.STD_Vol].iloc[-1], color='red')

        last_value = round(input_df[VolType.STD_Vol].iloc[-1], 2)
        percentile = round(stats.percentileofscore(input_df[VolType.STD_Vol], last_value), 2)

        # 添加注释：当前std_vol(20)和当前std_vol(20)所处的分位数
        ax3.annotate("[{}]\n标准差波动率={}\n分位数={}%".format(end_date, last_value, percentile),
                     xy=(last_value, 25),
                     xycoords='data', xytext=(15, 30),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->'))

        filepath3 = os.path.join(self._job_folder, f"{sec_id}_std_vol_his_{window}.png")
        fig3.savefig(filepath3, bbox_inches='tight')
        plt.close("all")

        return [filepath2, filepath3]

    def hist_vol_analysis(self, ud_code, bk_year=1):
        start_date = datetime.now() - timedelta(days=365 * bk_year + max(self._window_list))
        df = RQData.get_history_bar(ud_code, start_date, TIME.live_today()).loc[:, :, ud_code].sort_index()
        img_url_list = []
        his_vol_df = pd.DataFrame()
        for window in self._window_list:
            new_img_list, new_his_vol_df = self.plot_his_vol(ud_code, df, window, bk_year)
            img_url_list.extend(new_img_list)
            new_his_vol_df.columns = [x + f'_{window}' for x in new_his_vol_df.columns]
            his_vol_df = pd.merge(his_vol_df, new_his_vol_df, how='outer', right_index=True, left_index=True)
            # his_vol_df = pd.concat([his_vol_df, new_his_vol_df], axis=0, sort=False)

        return img_url_list, his_vol_df

    def plot_heatmap(self, sec_id, input_df):
        """
        画出最近20个交易日的波动率热力图
        :return:图片绝对路径
        """
        input_df = input_df.copy()

        df = input_df.iloc[-19:, :]
        df = df.sort_index(ascending=False)
        last_update = df.index[0]

        fig, ax = plt.subplots(figsize=(16, 8))
        # fig, ax = plt.subplots()
        df.rename(columns=lambda x: str(x).replace("_vol", ''), inplace=True)
        # 这里设置波动率10%为中心，为不同标的提供一个统一标准
        # sns.set(font_scale=1.5)
        # sns.set_style({"font.sans-serif": ['SimHei', 'Microsoft YaHei']})
        # sns.set_context('talk', font_scale=1.0)
        sns_plot = sns.heatmap(
            ax=ax, data=df, annot=True, fmt=".2f", square=False, center=10, cmap="YlGnBu",  # font_size=10
        )

        sns_plot.set_title("【{}】历史波动率分析：截止{}".format(sec_id, last_update))
        fig = sns_plot.get_figure()
        filepath = os.path.join(self._job_folder, f"{sec_id}_heatmap.png")
        fig.savefig(filepath, bbox_inches='tight')
        plt.close('all')
        return [filepath]

    def merge_images(self, image_list, common_str=None, prefix=None):
        prefix = '' if prefix is None else f'{prefix}_'
        if common_str is None:
            return [concat_img(image_list, self._job_folder, f"{prefix}{common_str}_merged")]
        else:
            images_to_be_merge = []
            for x in image_list:
                if common_str in str(x):
                    images_to_be_merge.append(x)
            return concat_img(images_to_be_merge, self._job_folder, f"{prefix}{common_str}_merged")

    def merge_img(self, img_list, prefix):
        img_list = [str(x).replace("\\", "/").replace("//", "/") for x in img_list]
        i_vix_img_list = [x for x in img_list if 'iVIX' in str(x)]
        vol_20_img_list = [x for x in img_list if 'his_vol_20' in str(x)]
        vol_40_img_list = [x for x in img_list if 'his_vol_40' in str(x)]
        vol_60_img_list = [x for x in img_list if 'his_vol_60' in str(x)]
        vol_gk_std_list = [x for x in img_list if 'gk-std' in str(x)]
        vol_hist_list = [x for x in img_list if 'std_vol_his' in str(x)]
        vol_heatmap_list = [x for x in img_list if 'heatmap' in str(x)]

        # 直接在HTML邮件中链接图片地址是不行的。引文大部分邮件服务商都会自动屏蔽带有外链的图片，因为不知道这些链接是否指向恶意网站。
        # 只需要在HTML中通过引用src="cid:0"就可以把附件作为图片嵌入了。如果有多个图片，给它们依次编号，然后引用不同的cid:x即可。

        concat_img_list = []
        # cur_date = datetime.now().date()
        concat_img_list += concat_img(i_vix_img_list, self._job_folder, f"{prefix}_merged_iVIX")
        concat_img_list += concat_img(vol_20_img_list, self._job_folder, f"{prefix}_merged_Vol_20")
        concat_img_list += concat_img(vol_40_img_list, self._job_folder, f"{prefix}_merged_Vol_40")
        concat_img_list += concat_img(vol_60_img_list, self._job_folder, f"{prefix}_merged_Vol_60")
        concat_img_list += concat_img(vol_gk_std_list, self._job_folder, f"{prefix}_merged_Vol_diff")
        concat_img_list += concat_img(vol_hist_list, self._job_folder, f"{prefix}_merged_Vol_his")
        concat_img_list += concat_img(vol_heatmap_list, self._job_folder, f"{prefix}_merged_Vol_heatmap")

        return concat_img_list

    def load(self, code=None, start=None, end=None, vol_type=None, window=None, *args, **kwargs):
        if vol_type == VolType.IVix:
            query = Query(IVix)
            table = IVix
        else:
            query = Query(Volatility)
            table = Volatility

        if start is not None:
            query = query.filter(table.AsOfTS >= pd.to_datetime(start).replace(hour=0, minute=0).to_pydatetime())

        if end is not None:
            query = query.filter(table.AsOfTS <= pd.to_datetime(end).replace(hour=23, minute=59).to_pydatetime())

        if vol_type is not VolType.IVix and window is not None:
            query = query.filter(table.Window == window)

        if code is not None:
            if vol_type == VolType.IVix:
                query = query.filter(table.Ud_Code == code)
            else:
                query = query.filter(table.Code == code)
        return pd.read_sql(query.statement, self._eng)

    def update(self, ivix_freq='1d', risk_free=0.03, verbose=False, *args, **kwargs):
        assert self._eng is not None
        s = sessionmaker(self._eng)()
        latest_ivix_ts = Query(func.max(IVix.AsOfTS), s).one_or_none()[0]
        latest_vol_ts = Query(func.max(Volatility.AsOfTS), s).one_or_none()[0]
        s.close()
        if latest_ivix_ts is None:
            if ivix_freq == '1d':
                self.restore('2016-01-01', TIME.live_today(), ivix_freq=ivix_freq, flush=False, verbose=True)
            elif ivix_freq == '1m':
                self.restore('2019-12-23', TIME.live_today(), ivix_freq=ivix_freq,
                             flush=False, verbose=True)
            else:
                raise NotImplementedError
        else:
            self.restore(TIME.live_today(), TIME.live_today(), ivix_freq=ivix_freq, verbose=verbose)

    def restore(self, start, end, ivix_freq='1d', risk_free=0.03, flush=False, verbose=False, **kwargs):
        assert self._eng is not None

        start = pd.to_datetime(start).to_pydatetime()
        end = pd.to_datetime(end).to_pydatetime()

        if flush:
            truncate_table(IVix, self._eng)
            truncate_table(Volatility, self._eng)

        # Base.metadata.create_all(self._eng)
        s = sessionmaker(self._eng)()
        s.query(IVix).filter(IVix.AsOfTS >= start.replace(hour=0, minute=0)). \
            filter(IVix.AsOfTS <= end.replace(hour=23, minute=59)).delete()
        s.query(OptionBasis).filter(OptionBasis.AsOfTs >= start.replace(hour=0, minute=0)). \
            filter(OptionBasis.AsOfTs <= end.replace(hour=23, minute=59)).delete()
        s.query(Volatility).filter(Volatility.AsOfTS >= start.replace(hour=0, minute=0)). \
            filter(Volatility.AsOfTS <= end.replace(hour=23, minute=59)).delete()

        s.commit()

        real_vol_hist = self._vol_section_cal(pd.to_datetime(start) - pd.DateOffset(years=1), end)

        for dt in RQData.get_trading_dates(start, end):
            self._load_option_info(dt)
            self._load_ud_quote(dt, ivix_freq)
            self._load_option_quote(dt, ivix_freq)
            self._update_option_strikes()
            self._update_option_ttm(dt, dt + pd.Timedelta(days=365))
            if ivix_freq == '1d':
                for ud in self._ivix_ud:
                    self._option_basis_snapshot_update(dt, s, risk_free, ud, verbose=verbose)
                    self._ivix_snapshot_update(dt, s, risk_free, ud, verbose)
            else:
                for ts in RQData.get_trading_timeline('000001.XSHG', dt, dt, freq=ivix_freq):
                    if ts > TIME.now():
                        break
                    for ud in self._ivix_ud:
                        self._option_basis_snapshot_update(ts, s, risk_free, ud, verbose=verbose)
                        self._ivix_snapshot_update(ts, s, risk_free, ud, verbose)
            if TIME.today() > dt.date() or TIME.now().hour >= 15:
                vol_snapshot = real_vol_hist.loc[dt, :].rename(JarvisFields.VolValue.Eng).reset_index().rename(columns={
                    'level_0': JarvisFields.Code.Eng,
                    'level_1': JarvisFields.VolType.Eng,
                    'level_2': JarvisFields.Window.Eng,
                })
                self._vol_snapshot_update(dt, s, vol_snapshot, verbose)

            s.commit()
        s.close()

    def _load_option_info(self, date):
        all_info = RQData.get_security_series(InstrumentType.Option.value, date)
        needed_info = all_info[
            all_info[InstrumentAttr.UnderlyingSymbol.value].isin(self._ivix_ud)  # &
            # all_info[InstrumentAttr.OptionName.value].apply(lambda x: 'A' not in x)
        ].set_index(InstrumentAttr.Code.value)

        if needed_info.groupby([InstrumentAttr.UnderlyingSymbol.value])[InstrumentAttr.MaturityDate.value]. \
                apply(lambda ser: ser.unique().shape[0]).min() <= 2:
            needed_info = all_info[
                all_info[InstrumentAttr.UnderlyingSymbol.value].isin(self._ivix_ud)
            ].set_index(InstrumentAttr.Code.value)

        day_bar = RQData.get_history_bar(list(needed_info.index), start=date, end=date)
        if day_bar is not None:
            multiplier = day_bar.loc[InstrumentAttr.ContractMultiplier.value, date, needed_info.index]
        else:
            wind_mapping = {
                CodeRegulator.rq_to_wind(idx, needed_info.loc[idx, 'exchange']): idx for idx in needed_info.index
            }
            multiplier = Wind.get_history_bar(
                wind_mapping.keys(), start=date, end=date, fields=['contractmultiplier']
            ).iloc[0, 0, :]
            multiplier.index = [wind_mapping[x] for x in multiplier.index]

        self._option_info = needed_info.loc[multiplier[multiplier.isin(self._atm_multipliers)].index, :]

    def _load_ud_quote(self, date, freq):
        self._ud_quote = RQData.get_history_bar(
            codes=self._ivix_ud,
            start=date.replace(hour=0, minute=0),
            end=date.replace(hour=23, minute=59),
            freq=freq
        )
        if self._ud_quote is None:
            raise RuntimeError('Acquire empty data of underlying from RQ .')

    def _load_option_quote(self, date, freq):
        if self._option_info is None:
            self._load_option_info(date)

        self._option_quote = RQData.get_history_bar(
            codes=self._option_info.index,
            start=date.replace(hour=0, minute=0),
            end=date.replace(hour=23, minute=59),
            freq=freq,
        )
        if self._option_quote is None:
            instruments = RQData.get_instrument(self._option_info.index)
            wind_code_mapping = {
                CodeRegulator.rq_to_wind(i.code, i.exchange): i.code for i in instruments
            }
            quote = Wind.get_history_bar(
                codes=list(wind_code_mapping.keys()),
                start=date.replace(hour=0, minute=0),
                end=date.replace(hour=23, minute=59),
                freq=freq,
                fields=['close'],
            )
            quote.minor_axis = [wind_code_mapping[x] for x in quote.minor_axis]
            self._option_quote = quote

    def _update_option_strikes(self):
        for ud_code in self._ivix_ud:
            self._option_strike_lib[ud_code] = []
            option_info = self._option_info.groupby([InstrumentAttr.UnderlyingPrdCode.value]).get_group(ud_code)
            expiration_list = sorted(set(pd.to_datetime(option_info[InstrumentAttr.DeListedDate.value])))
            for exp_dt in expiration_list:
                temp_info = option_info[option_info[InstrumentAttr.MaturityDate.value] == f'{exp_dt:%Y-%m-%d}']
                call_strip = temp_info.loc[
                    temp_info[InstrumentAttr.OptionType.value] == OptionType.Call.value,
                    [InstrumentAttr.StrikePrice.value]
                ]

                put_strip = temp_info.loc[
                    temp_info[InstrumentAttr.OptionType.value] == OptionType.Put.value,
                    [InstrumentAttr.StrikePrice.value]
                ]
                if call_strip.shape[0] > 0 and put_strip.shape[0] > 0:
                    self._option_strike_lib[ud_code].append((call_strip, put_strip))

    def _update_option_ttm(self, dt, extend_dt):
        if self._trading_dates is None:
            self._trading_dates = RQData.get_trading_dates(dt, extend_dt)
        else:
            for date in RQData.get_trading_dates(self._trading_dates[-1] + pd.Timedelta(days=1), extend_dt):
                self._trading_dates.append(date)

        for ud_code, option_info in self._option_info.groupby([InstrumentAttr.UnderlyingPrdCode.value]):
            self._option_ttm_lib[ud_code] = []
            for exp_dt in sorted(set(pd.to_datetime(option_info[InstrumentAttr.MaturityDate.value]))):
                if exp_dt < dt:
                    continue
                self._option_ttm_lib[ud_code].append((
                    exp_dt,
                    self._trading_dates.index(exp_dt) - self._trading_dates.index(dt)
                ))

    def _ivix_snapshot_update(self, ts, sess, risk_free=0.03, ud_code='510050.XSHG', verbose=False):

        if ud_code not in self._option_info[InstrumentAttr.UnderlyingSymbol.value].unique():
            ivix = np.nan
        else:
            ivix = self._ivix_cal(ts, risk_free, ud_code)

        if not np.isnan(ivix):
            if ts.hour == 0:
                ts = ts.replace(hour=15)
            ivix_obj = IVix(AsOfTS=ts.to_pydatetime(), Ud_Code=ud_code, IVIX=float(ivix))
            sess.add(ivix_obj)

        if verbose:
            click.echo(f'{ts:%Y-%m-%d %H:%M:%S}: {ud_code}/{VolType.IVix}/{ivix:.4f}')

    def _option_basis_snapshot_update(
            self, ts, sess, risk_free=0.03, ud_code='510050.XSHG', annual_factor=244, verbose=False
    ):
        ud_price = self._ud_quote.loc['close', ts, ud_code]

        for i in range(len(self._option_ttm_lib[ud_code])):
            maturity, ttm = self._option_ttm_lib[ud_code][i]
            ts_frac = self.ts_frac_cal(ts)
            try:
                call_strikes, put_strikes = self._option_strike_lib[ud_code][i]
            except IndexError:
                continue
            call_strip = call_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                             set_index(InstrumentAttr.StrikePrice.value).loc[:, 'close']
            put_strip = put_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                            set_index(InstrumentAttr.StrikePrice.value).loc[:, 'close']
            atm_strike = (call_strip - put_strip).dropna().abs().argmin()
            for k in call_strip.index:
                fwd = Calculator.forward_price(
                    call_price=call_strip[k],
                    put_price=put_strip[k],
                    strike=k,
                    days_to_maturity=ttm + ts_frac,
                    risk_free=risk_free,
                    annual_factor=annual_factor
                )
                basis = fwd - ud_price
                sess.add(OptionBasis(
                    AsOfTs=ts.to_pydatetime(),
                    Ud_Code=ud_code,
                    Maturity=maturity.date(),
                    MaturityLabel=i,
                    TTM=float(ttm + ts_frac),
                    Strike=k,
                    ForwardPx=float(fwd),
                    IsATM=k == atm_strike,
                    Basis=float(basis),
                ))

                if verbose:
                    click.echo(f'{ts:%Y-%m-%d %H:%M:%S}: {ud_code}/{i}/{k}{"*" if k == atm_strike else ""}/{basis:.4f}')

    def _vol_section_cal(self, start, end):

        data = RQData.get_history_bar(self._code_list, start, end)
        real_vol_holder = {}
        for code, window in product(self._code_list, self._window_list):
            real_vol_holder.update({
                (code, VolType.STD_Vol, window): self.std_vol_cal(window=window, data=data.loc[:, :, code]) * 100,
                (code, VolType.PK_Vol, window): self.pk_vol_cal(window=window, data=data.loc[:, :, code]) * 100,
                (code, VolType.GK_Vol, window): self.gk_vol_cal(window=window, data=data.loc[:, :, code]) * 100,
            })

        return pd.DataFrame(real_vol_holder)

    def _vol_snapshot_update(self, dt, sess, snapshot=None, verbose=False):

        if snapshot is None:
            raise NotImplementedError

        for idx in snapshot.index:
            code, vol_type, window, vol_value = snapshot.loc[
                idx,
                [JarvisFields.Code.Eng, JarvisFields.VolType.Eng, JarvisFields.Window.Eng, JarvisFields.VolValue.Eng]
            ]

            sess.add(Volatility(
                AsOfTS=dt.replace(hour=15).to_pydatetime(),
                Code=code,
                Window=int(window),
                VolType=vol_type,
                VolValue=float(vol_value))
            )
            if verbose:
                click.echo(f'{dt:%Y-%m-%d}: {code}/{vol_type}/{window}/{vol_value:.4f}')

    def _ivix_cal(self, ts, risk_free, ud_code='510050.XSHG'):

        # near_call, near_put, near_expiration, next_call, next_put, next_expiration = \
        #     self._get_near_next_option_quote(ts, ud_code)
        (_, near_ttm), (_, next_ttm) = self._option_ttm_lib[ud_code][:2]

        near_call_strikes, near_put_strikes = self._option_strike_lib[ud_code][0]
        next_call_strikes, next_put_strikes = self._option_strike_lib[ud_code][1]

        near_call_strip = near_call_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                              set_index(InstrumentAttr.StrikePrice.value).iloc[:, -1]
        near_put_strip = near_put_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                             set_index(InstrumentAttr.StrikePrice.value).iloc[:, -1]
        next_call_strip = next_call_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                              set_index(InstrumentAttr.StrikePrice.value).iloc[:, -1]
        next_put_strip = next_put_strikes.join(self._option_quote.loc[['close'], ts, :], how='left'). \
                             set_index(InstrumentAttr.StrikePrice.value).iloc[:, -1]

        idx_of_cur_date = self._trading_dates.index(ts.replace(hour=0, minute=0))

        remain_ratio = self.ts_frac_cal(ts)

        near_vix = Calculator.vix_cal(
            near_call_strip,
            near_put_strip,
            days_to_maturity=near_ttm + remain_ratio,
            annual_factor=244,
            risk_free=risk_free
        )

        idx_of_30_later = bisect.bisect_left(
            self._trading_dates, ts.replace(hour=0, minute=0) + pd.Timedelta(days=30)
        )
        days_of_30_later = idx_of_30_later - idx_of_cur_date

        if near_ttm > days_of_30_later:
            ivix = 100 * near_vix
        else:

            next_vix = Calculator.vix_cal(
                next_call_strip,
                next_put_strip,
                days_to_maturity=next_ttm + remain_ratio,
                annual_factor=244,
                risk_free=risk_free
            )

            near_weight = (next_ttm - days_of_30_later) / \
                          (next_ttm - near_ttm)
            next_weight = (days_of_30_later - near_ttm) / \
                          (next_ttm - near_ttm)
            ivix = 100 * np.sqrt(
                ((near_ttm + remain_ratio) * near_vix ** 2 * near_weight +
                 (next_ttm + remain_ratio) * next_vix ** 2 * next_weight) / days_of_30_later
            )

        return ivix

    @classmethod
    def ts_frac_cal(cls, ts):
        # 4 trading hours per day
        if ts.time() < time(hour=9, minute=30):
            frac = 1.0
        elif time(hour=9, minute=30) <= ts.time() <= time(hour=11, minute=30):
            frac = 1 - (ts - ts.replace(hour=9, minute=30)).seconds / (4 * 60 * 60)
        elif time(hour=11, minute=30) < ts.time() < time(hour=13):
            frac = 0.5
        elif time(hour=13) <= ts.time() <= time(hour=15):
            frac = (ts.replace(hour=15, minute=0) - ts).seconds / (4 * 60 * 60)
        else:
            frac = 0.0

        return frac

    @classmethod
    def std_vol_cal(cls, code=None, date=None, window=None, annual_factor=244, data=None):
        if data is None:
            close = RQData.get_history_bar(code, date - timedelta(days=365), date).loc[
                    'close', :, code].iloc[-(window + 1):]
        else:
            close = data['close']
        log_yield = close.apply(np.log).diff()
        return log_yield.rolling(window=window).std() * np.sqrt(window / (window - 1) * annual_factor)

    @classmethod
    def pk_vol_cal(cls, code=None, date=None, window=None, annual_factor=244, data=None):

        if data is None:
            df = RQData.get_history_bar(code, date - timedelta(days=365), date).loc[
                 :, :, code].iloc[-(window + 1):]
        else:
            df = data

        # 2020.2.3日出现异常回撤，需特殊处理
        if pd.to_datetime('2020-02-03') in df.index and pd.to_datetime('2020-01-23') in df.index:
            df.loc[pd.to_datetime('2020-02-03'), 'high'] = \
                df.loc[pd.to_datetime('2020-01-23'), 'close']

        log_hl_square = (df['high'] / df['low']).apply(np.log) ** 2
        return np.sqrt(log_hl_square.rolling(window=window).mean() / (4 * np.log(2)) * annual_factor)

    @classmethod
    def gk_vol_cal(cls, code=None, date=None, window=None, annual_factor=244, data=None):

        if data is None:
            df = RQData.get_history_bar(code, date - timedelta(days=365), date).loc[
                 :, :, code].iloc[-(window + 1):]
        else:
            df = data

        # 2020.2.3日出现异常回撤，需特殊处理
        if pd.to_datetime('2020-02-03') in df.index and pd.to_datetime('2020-01-23') in df.index:
            df.loc[pd.to_datetime('2020-02-03'), 'high'] = \
                df.loc[pd.to_datetime('2020-01-23'), 'close']

        log_hl_square = (df['high'] / df['low']).apply(np.log) ** 2
        log_cc_square = (df['close'] / df['close'].shift(1)).apply(np.log) ** 2

        part1 = 0.5 * log_hl_square.rolling(window=window).mean()
        part2 = (2 * np.log(2) - 1) * log_cc_square.rolling(window=window).mean()
        return ((part1 - part2) * annual_factor).apply(np.sqrt)


if __name__ == '__main__':
    RQData.init()
    EnginePointer.renew('finance_database')
    # Wind.init()
    # EnginePointer.renew('dev')
    # from jarvis.jobs.vol_analysis import VolAnalysis
    # JobManager.install_job('vol_analysis', VolAnalysis, JobType.Module, activate=True)
    job = VolAnalysis()
    # manager = JobManager(alert=True)
    # job.register_event(event_bus=manager.event_bus, job_uuid=None, debug=False)
    # manager.event_bus.event_queue_reload(EVENT.PM0400)
    # manager.event_bus.sequential_publish()
    # job.run(ivix_freq='1m',hist_bkwd_year=5, verbose=True, debug=False)
    job.restore(datetime.today().strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d'), '1m', verbose=True)
