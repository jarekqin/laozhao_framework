import pandas as pd
import os

from omk.core.orm_db import EnginePointer
from basic_indicators.sql_toolkits.data_db import NORTHTOSOUTHINFLOW, NORTHTOSOUTHNETBUY, SOUTHTONORTHINFLOW, \
    SOUTHTONORTHNETBUY


def read_file_df(read_file_path, read_file_type='csv', encoding='utf8'):
    if read_file_type == 'csv':
        return pd.read_csv(read_file_path, encoding=encoding)
    elif read_file_type == 'excel':
        return pd.read_excel(read_file_path, encoding=encoding)
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


def organise_df_and_save(read_file_path, table_name, picker_name='finance_database', encoding='utf8',
                         read_file_type='csv'):
    if picker_name is None:
        engine = EnginePointer.get_engine()
    else:
        engine = EnginePointer.picker(picker_name)

    for file in read_file_path:
        temp_df = read_file_df(file, read_file_type, encoding)
        if temp_df.shape[0] == 0:
            print('%s data is empty!' % file)
            continue
        try:
            temp_df.to_sql(table_name.table_name(), schema=table_name.schema(), con=engine, if_exists='append',
                           index=False)
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    files_path = 'E:\\老赵分析框架\\north_south_1min_data'
    file_paths = get_files_path(files_path)
    organise_df_and_save(files_path, 'utf8', 'csv')
