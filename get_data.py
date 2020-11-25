import pandas as pd
from influxdb import InfluxDBClient

class DB_api():

    def __init__(self, ip, port):
        self.dbname = 'debs2020'
        self.client = InfluxDBClient(ip, port, '', '', self.dbname)

        #获取数据表第一批数据的开始时间
        res_first = self.client.query('select * from vol_cur limit 1')
        for item in res_first.get_points():
            self.time_start = item['time']

    def get_batch(self, batchCounter):
        
        response = {}

        # select * from vol_cur where "tag" = '4'

        # 获取本批数据的开始时间
        
        bat_time_start = batchCounter*1000+self.time_start

        print(bat_time_start)

        # response['x'] = self.client.query('select voltage,current from vol_cur where time <= 1606148216276000000 ')
        
        

# PATH = 'E:\download\dataset_1\dataset_1_20200311\in1.csv'
# df = pd.DataFrame(pd.read_csv(PATH, header=None))
# df = df.loc[:, 1:2]
# df.columns = ['voltage', 'current']

# def get_batch(flag):
#     return df.iloc[flag*1000:(flag*1000+1000)]

if __name__ == "__main__":
    my_influxclient = DB_api('localhost', 8086)
    my_influxclient.get_batch(0)
