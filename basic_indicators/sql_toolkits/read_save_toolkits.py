import pandas as pd
import os

from omk.core.orm_db import EnginePointer
from basic_indicators.sql_toolkits.data_db import NORTHTOSOUTHINFLOW, NORTHTOSOUTHNETBUY, SOUTHTONORTHINFLOW, \
    SOUTHTONORTHNETBUY


def read_file_df(read_file_path, read_file_type='csv', encoding='utf-8'):
    if read_file_type == 'csv':
        temp_df=pd.read_csv(read_file_path, encoding=encoding)
        temp_df.datetime=pd.to_datetime(temp_df.datetime)
        return temp_df
    elif read_file_type == 'excel':
        temp_df=pd.read_excel(read_file_path, encoding=encoding)
        temp_df.datetime = pd.to_datetime(temp_df.datetime)
        return temp_df
    else:
        raise TypeError


def get_files_path(read_file_path):
    files_paths = []
    for root, dirs, files in os.walk(read_file_path):
        if len(dirs) == 0:
            for file in files:
                files_paths.append(os.path.join(root, file))
    if len(files_paths) > 0:
        return files_paths
    else:
        return []


def organise_df_and_save(read_file_path, table_name, picker_name='finance_database', encoding='utf-8',
                         read_file_type='csv'):
    if picker_name is None:
        engine = EnginePointer.get_engine()
    else:
        engine = EnginePointer.picker(picker_name)

    temp_df = read_file_df(read_file_path, read_file_type, encoding)
    if temp_df.shape[0] == 0:
        print('%s data is empty!' % file)
    try:
        temp_df.to_sql(table_name.table_name(), schema=table_name.schema(), con=engine, if_exists='append',
                       index=False)
    except Exception as e:
        print(e)


def save_to_sql(df,table_name,con,index_col=False):
    try:
        df.to_sql(table_name.table_name(), schema=table_name.schema(), con=con, if_exists='append',
                       index=index_col)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    files_path = 'E:\\老赵分析框架\\north_south_1min_data'
    file_paths = get_files_path(files_path)
    print(file_paths)
    for file in file_paths:
        if 'net_buy' not in file:
            if 'n2s' in file:
                organise_df_and_save(file,table_name=NORTHTOSOUTHINFLOW, encoding='utf-8', read_file_type='csv')
            else:
                organise_df_and_save(file, table_name=SOUTHTONORTHINFLOW,encoding='utf-8', read_file_type='csv')
        else:
            if 'n2s' in file:
                organise_df_and_save(file, table_name=NORTHTOSOUTHNETBUY, encoding='utf-8', read_file_type='csv')
            else:
                organise_df_and_save(file, table_name=SOUTHTONORTHNETBUY, encoding='utf-8', read_file_type='csv')
    # organise_df_and_save(files_path, 'utf8', 'csv')
