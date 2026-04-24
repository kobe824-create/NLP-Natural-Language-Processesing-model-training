import pandas as pnd

data = pnd.read_csv('./dataset/Bengaluru_House_Data.csv')
df = pnd.DataFrame(data)


# pnd.set_option('display.max_rows', None)
# print(df.head(50))
# print(df.info())
# print(df.duplicated().sum())
print(df.describe())