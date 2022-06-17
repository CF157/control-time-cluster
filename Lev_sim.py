import Levenshtein as Lev
import pandas as pd
import numpy as np
import pickle

def String_trace(dgroup):#把迹用字符串表示
    str_t=[]
    for cid,df in dgroup:
        acts="".join(df.iloc[:,1])
        str_t.append(acts)

    return str_t

def cal_sim(str_list):#使用Levenshtein距离计算迹的距离
    str_size=len(str_list)
    sim_jz = np.zeros((str_size, str_size))
    for i,str1 in zip(range(str_size),str_list):
        for j,str2 in zip(range(str_size),str_list):
            if i==j:
                continue

            dis=Lev.distance(str1,str2)
            fenmu=dis+1
            sim_jz[i][j]=1/fenmu
    return sim_jz

# data = pd.read_excel('../precess/Log_data/PrepaidTravelCost.xlsx')
data = pd.read_excel('../precess/Log_data/BPI_Challenge_2013_closed_problems.xlsx')
def convert2str(x):
    x = str(x)
    return x
data['act']=data['act'].apply(convert2str)
dg = data.groupby(['case_id'])

str_t=String_trace(dg)
# print('字符串迹',str_t)
print(len(str_t))
sim=cal_sim(str_t)
print(sim)
print(len(sim))

preprocessed_data_name = 'Lev_sim.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(sim, f, protocol=2)

print('Lev相似性，写入成功！！！')

# str1='hello'
# str2='world'
# dis=Lev.distance(str1,str2)
# print(dis)

