import pandas as pd
import numpy as np
import pickle

#用变体文件跑
# data = pd.read_excel('v01.xlsx')
data=pd.read_excel('../precess/Log_data/Log_v/PrepaidTravelCost_v.xlsx')
dg = data.groupby(['case_id'])
v_len=len(dg)#变体的数量
print('变体数量',v_len)
thre=0.9#置信度阈值

def convert2str(x):
    x = str(x)
    return x
data['act']=data['act'].apply(convert2str)

def f(df_group):#找出各个活动的前置活动集
    act_he_dic = {}
    for c_id,gp in df_group:
        index_dic={}
        acts = list(gp.iloc[:, 1])
        for a_i,act in zip(range(len(acts)),acts):
            key=[k for k in index_dic.keys()]
            if act not in key:
                index_dic[act]=[a_i]
            else:
                index_dic[act].append(a_i)

        for ky, vl in index_dic.items():  # 找出每个活动的前置活动集
            t_act_b = []
            t_i = 0
            s_t = 0
            for v in vl:
                if s_t == 0:  # 判断是否是第一次执行循环（如果索引不为0，则从0开始）
                    start = 0
                else:  # 不是第一次执行，则起始索引不包括当前活动
                    start = t_i + 1
                if v == 0:
                    s_t = 1
                    continue
                s_t = 1
                arr = np.array(acts)
                xiao_ji = list(arr[start:v])
                t_act_b = t_act_b + xiao_ji
                t_i = v
            ahd_key = [a_k for a_k in act_he_dic.keys()]
            if ky in ahd_key:
                # if len(t_act_b) > 0:
                    act_he_dic[ky].append(t_act_b)
            else:
                # if len(t_act_b) > 0:
                    act_he_dic[ky] = [t_act_b]
    return act_he_dic

act_he_dic=f(dg)
# print(act_he_dic)

def sup_conf(he_dic,yuzhi):
    s_c={}
    for key,val in he_dic.items():#val=[[],[],[]]
        act_lb=[]
        for v in val:#v=[],统计每个活动的前置活动的活动类别
            if len(v)<1:
                continue
            for act in v:
                if act not in act_lb:
                    act_lb.append(act)
        # print(act_lb)
        act_num={}
        for act2 in act_lb:#统计每个活动的前置活动的活动次数
            count=0
            for v2 in val:
                if len(v2)<1:
                    continue
                if act2 in v2:
                    count=count+1
            act_num[act2]=count
        # print(act_num)
        g_l_act=[]#key的关联活动的集合
        a_l=len(val)#每个key在变体中出现的次数
        for key2,val2 in act_num.items():
            temp=val2/a_l
            if temp>yuzhi:
                g_l_act.append(key2)
        s_c[key]=g_l_act
    return s_c


s_c=sup_conf(act_he_dic,thre)
print(s_c)

def get_px(a_h_dic,sc):
    act_fz_dic={}
    for key,value in a_h_dic.items():
        gl_act=sc[key]
        new_value =[]
        for ind,vl in zip(range(len(value)),value):                #value=[['b'], ['b', 'c']],反转，去除同一列表相同活动
            vl.reverse()
            temp=[]
            for act in vl:                                     #vl=['b', 'c']
                if act not in gl_act:
                    continue
                if act not in temp:
                    temp.append(act)
            new_value.append(temp)

        dic2={}
        for val in new_value:#统计每个new_value中的val第一个活动的数量
            key2=[k for k in dic2.keys()]
            if len(val)<1:
                continue
            act2=val[0]
            if act2 not in key2:
                dic2[act2]=1
            else:
                dic2[act2]=dic2[act2]+1

        if len(dic2)<1:#判断字典是否是空的
            act_fz_dic[key] = []
            continue

        temp_key='a'
        temp_value=0
        for key3,val3 in dic2.items():#获取字典中出现次数最大的活动
            if val3>temp_value:
                temp_key=key3
                temp_value=val3

        act_px=[temp_key]#调整活动顺序，把出现次数最多的活动放到第一个位置
        for act3 in gl_act:
            if act3 not in act_px:
                act_px.append(act3)


        # print(dic2)
        act_fz_dic[key]=act_px
        # act_fz_dic[key]=new_value
    return act_fz_dic

act_px_dic=get_px(act_he_dic,s_c)
print(act_px_dic)

preprocessed_data_name = 'time_dic_yc.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(act_px_dic, f, protocol=2)
print('写入成功！！！')
