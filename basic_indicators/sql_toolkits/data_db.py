from sqlalchemy import Column, FLOAT,VARCHAR, DATETIME, TIME,BigInteger

from basic_indicators.sql_toolkits.sql_base import Base

class _SchemaBase(object):
    __tablename__ = NotImplemented
    __table_args__ = {'schema': 'north_south_data'}

    Key = Column('key', BigInteger, autoincrement=True, primary_key=True)

    @classmethod
    def table_name(cls):
        return cls.__tablename__

    @classmethod
    def schema(cls):
        return cls.__table_args__['schema']



class NORTHTOSOUTHINFLOW(_SchemaBase,Base):
    __tablename__ = 'n2s_1min'
    DATATIME=Column('datetime',DATETIME,comment='时间',index=True)
    HKSHNETINFLOW=Column('hk_sh_net_in_flow',FLOAT,comment='港沪通净流入资金')
    HKSHREMAINED=Column('hk_sh_remained',FLOAT,comment='港沪通剩余可用资金总量')
    HKSZNETINFLOW=Column('hk_sz_net_in_flow',FLOAT,comment='港深通净流入资金')
    HKSZREMAINED=Column('hk_sz_remained',FLOAT,comment='港深通剩余可用资金总量')
    SOUTHINFLOW=Column('southen_in_flow',FLOAT,comment='南向净流入资金')


class SOUTHTONORTHINFLOW(_SchemaBase,Base):
    __tablename__ = 's2n_1min'
    DATATIME=Column('datetime',DATETIME,comment='时间',index=True)
    HKSHNETINFLOW=Column('sh_net_in_flow',FLOAT,comment='沪港通净流入资金')
    HKSHREMAINED=Column('sh_remained',FLOAT,comment='沪港通剩余可用资金总量')
    HKSZNETINFLOW=Column('sz_net_in_flow',FLOAT,comment='深港通净流入资金')
    HKSZREMAINED=Column('sz_remained',FLOAT,comment='深港通剩余可用资金总量')
    NORTHTHINFLOW=Column('northen_in_flow',FLOAT,comment='北向净流入资金')


class NORTHTOSOUTHNETBUY(_SchemaBase,Base):
    __tablename__ = 'n2s_1min_net_buy'
    DATATIME=Column('datetime',DATETIME,comment='时间',index=True)
    HKSHBUY=Column('n2s_hk_sh_buy',FLOAT,comment='港沪通净买入')
    HKSHREMAINED=Column('n2s_hk_sz_buy',FLOAT,comment='港深通净买入')
    HKSHSELL=Column('n2s_hk_sh_sell',FLOAT,comment='港沪通净卖出')
    HKSZSELL = Column('n2s_hk_sz_sell', FLOAT, comment='港深通净卖出')
    SOUTHBUY=Column('south_buy', FLOAT, comment='南向净买入')
    SOUTHSELL = Column('south_sell', FLOAT, comment='南向净卖出')
    HKSHNET=Column('hk_sh_net',FLOAT,comment='港沪通净流入')
    HKSZNET = Column('hk_sz_net', FLOAT, comment='港深通净流入')
    SOUTHNET = Column('south_net', FLOAT, comment='南向净流入')


class SOUTHTONORTHNETBUY(_SchemaBase,Base):
    __tablename__ = 's2n_1min_net_buy'
    DATATIME=Column('datetime',DATETIME,comment='时间',index=True)
    SHBUY=Column('s2n_sh_buy',FLOAT,comment='沪港通净买入')
    SHREMAINED=Column('s2n_sz_buy',FLOAT,comment='深港通净买入')
    SHSELL=Column('s2n_sh_sell',FLOAT,comment='沪港通净卖出')
    SZSELL = Column('s2n_sz_sell', FLOAT, comment='深港通净卖出')
    NORTHBUY=Column('north_buy', FLOAT, comment='北向净买入')
    NORTHSELL = Column('north_sell', FLOAT, comment='北向净卖出')
    SHNET=Column('sh_net',FLOAT,comment='沪港通净流入')
    SZNET = Column('sz_net', FLOAT, comment='深港通净流入')
    NORTHNET = Column('north_net', FLOAT, comment='北向净流入')













if __name__=='__main__':
    from omk.core.orm_db import EnginePointer, build_table
    engine=EnginePointer.picker('finance_database')

    for table, renew in [
        (NORTHTOSOUTHINFLOW, True),
        (SOUTHTONORTHINFLOW, True),
        (NORTHTOSOUTHNETBUY, True),
        (SOUTHTONORTHNETBUY, True),

    ]:
        build_table(table, renew, engine, True)

