import pandas as pd
import numpy as np
import pickle

# data = pd.read_excel('../precess/Log_data/PrepaidTravelCost.xlsx')
# data = pd.read_excel('../precess/Log_data/Sepsis_Cases.xlsx')
data = pd.read_excel('../precess/Log_data/BPI_Challenge_2013_closed_problems.xlsx')
dg = data.groupby(['case_id'])


def tj_ac_lb(df_group):  # 统计活动类别
    act = []
    for i, g in df_group:
        acts = list(g.iloc[:, 1])
        for a in acts:
            if a not in act:
                act.append(a)
    return act


act_lb=tj_ac_lb(dg)
# act_lb = ['A', 'C', 'K', 'E', 'F', 'I', 'G', 'D', 'B', 'L', 'R', 'U', 'T', 'S', 'M', 'J', 'N', 'P', 'O', 'Q', 'H']
print(act_lb)
# print(len(act_lb))

def encode_t(df_group,actlb):
    t_jz=[]
    for i,g in df_group:
        t=list(g.iloc[:,1])
        xt=[0 for j in range(len(actlb))]
        for a in t:
            id=actlb.index(a)
            xt[id]=xt[id]+1
        t_jz.append(xt)
    return t_jz

en_t_jz=encode_t(dg,act_lb)
# print(en_t_jz)

def cal_sim(jz):
    l=len(jz)
    sim_jz = np.zeros((l, l))
    for i in range(len(jz)):
        for j in range(len(jz)):
            if i==j:
                continue
            else:
                ti=np.array(jz[i])
                tj=np.array(jz[j])
                cha=tj-ti
                f=cha*cha
                he=np.sum(f)
                gh=pow(he,0.5)
                sim_jz[i][j]=1/(gh+1)
    return sim_jz

sim_jz=cal_sim(en_t_jz)
print(sim_jz)
print(len(sim_jz))
preprocessed_data_name =  'sim_jz_act.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(sim_jz, f, protocol=2)

print('写入成功！！！')
