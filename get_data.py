import pandas as pd

PATH = 'E:\download\dataset_1\dataset_1_20200311\in1.csv'
df = pd.DataFrame(pd.read_csv(PATH, header=None))
df = df.loc[:, 1:2]
df.columns = ['voltage', 'current']

def get_batch(flag):
    return df.iloc[flag*1000:(flag*1000+1000)]

if __name__ == "__main__":
    get_batch()