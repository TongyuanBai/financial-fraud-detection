
"""
Created on Thu Sep 16 20:00:45 2021

@author: tongyuan bai
"""
import pandas as pd
import matplotlib.pyplot as plt

#统计频数
data = pd.read_csv("E:/essay/data/data1.csv")
num = data.groupby(["category","FLAG"]).size()

#判断数据是否为空
mis_val = data.drop(columns=['TICKER_SYMBOL', 'category','FLAG']).isnull().sum()
mis_val_percent =100 * data.drop(columns=['TICKER_SYMBOL', 'category','FLAG']).isnull().sum() / len(data)
print(mis_val_percent)
mis_val1 = mis_val_percent.tolist()

#缺失值可视化
plt.figure(figsize=(15,8))
x = list(range(352))
plt.bar(x,mis_val1,color="dodgerblue",label=False)
plt.xlabel("特征指标",fontsize=23)
plt.ylabel("缺失度",fontsize=23)
plt.show()

#对列名重命名
clear_col = mis_val_percent.to_frame()
clear_col.columns = ['value']

#缺失值填充
names1 = clear_col[clear_col['value']>=70]
drop_col_name1 = names1.index.tolist()
drop_col_name1

for i in drop_col_name1:
    data = data.drop(columns=[i])

names2 = clear_col[(clear_col['value'] < 70)&(clear_col['value'] >= 40)]
drop_col_name2 = names2.index.tolist()
drop_col_name2

for i in drop_col_name2:
    data[i] = data[i].fillna(0)

names3 = clear_col[(clear_col['value'] < 40)]
drop_col_name3 = names3.index.tolist()
drop_col_name3

for i in drop_col_name3:
    data[i] = data[i].fillna(data[i].mean())

#检查数据缺失情况
data.isnull().sum()

#处理后缺失值情况
mis_val_percent_process =100 * data.isnull().sum() / len(data)
print(mis_val_percent_process)

#缺失值可视化
mis_val2 = mis_val_percent_process.tolist()
plt.figure(figsize=(15,8))
x = list(range(240))
plt.bar(x,mis_val2,color="dodgerblue",label=False)
plt.xlabel("特征指标",fontsize=23)
plt.ylabel("缺失度",fontsize=23)
plt.show()

#导出文件
data.to_csv('F:/essay/data/data_clean.csv',index=False,header=True,encoding="utf_8_sig") 



