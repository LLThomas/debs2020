import pandas as pd
from influxdb import InfluxDBClient
import datetime
from pytz import timezone

class DB_api():

    def __init__(self, ip, port):
        self.dbname = 'debs2020'
        self.client = InfluxDBClient(ip, port, '', '', self.dbname)

    def df_convert(self, raw_dataset):
        # 生成vol和cur的dict
        dict_batch_data = {}
        list_vol = []
        list_cur = []
        for item in raw_dataset.get_points():
            list_vol.append(item['voltage'])
            list_cur.append(item['current'])
        dict_batch_data['voltage'] = list_vol
        dict_batch_data['current'] = list_cur
        # dict -> dataframe
        df_batch = pd.DataFrame.from_dict(dict_batch_data)
        return df_batch

    def get_batch(self, batchCounter):
        
        # 获取偏移量
        offset = 1000*batchCounter

        # 获取某个数据段的原始数据
        raw_dataset = self.client.query('select * from vol_cur limit 1000 offset %d'%offset)

        # raw_dataset -> dataframe
        df_batch = self.df_convert(raw_dataset)

        return df_batch

if __name__ == "__main__":
    my_influxclient = DB_api('localhost', 8086)
    my_influxclient.get_batch(0)
