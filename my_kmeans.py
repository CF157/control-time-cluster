import random


def adjust_center(sim_jz,dic):
    new_c=[]
    for key,val in dic.items():
        temp_sim=0
        temp_t=key
        for t1 in val:
            sum_sim=0
            for t2 in val:
                if t1==t2:
                    continue
                sum_sim=sum_sim+sim_jz[t1][t2]
            if sum_sim>temp_sim:
                temp_sim=sum_sim
                temp_t=t1
        new_c.append(temp_t)
    return new_c

def my_k(sim_jz,k,clear_t):
    t_len=len(clear_t)
    rand_num=[]
    while len(rand_num)<k:#随机选择k个点，当作中心点
        num = random.randint(0, t_len - 1)
        if num not in rand_num:
            rand_num.append(clear_t[num])
    # print(rand_num)
    flg=True
    # print(rand_num)
    temp_num=rand_num
    while flg:
        temp_cluster={}
        for n in temp_num:#初始化字典
            temp_cluster[n]=[]
        key=[k for k in temp_cluster.keys()]
        # print('key',key)
        # print('temp_num',temp_num)
        for ind in clear_t:#根据中心点划分迹
            if ind in temp_num:
                temp_cluster[ind].append(ind)
                continue
            max_sim=0
            temp_t=temp_num[0]
            for j in temp_num:
                sim=sim_jz[j][ind]
                if sim>max_sim:
                    max_sim=sim
                    temp_t=j
            # print('temp_t',temp_t)
            temp_cluster[temp_t].append(ind)

        new_center=adjust_center(sim_jz,temp_cluster)#调整中心点
        # print('dic', temp_cluster)
        # print('new_c',new_center)
        test=[f for f in new_center if f not in temp_num]
        if len(test)==0:
            flg=False
        else:
            temp_num=new_center

    clusters=[]
    for val in temp_cluster.values():
        clusters.append(val)

    return clusters

def tichu_cu(jz_size,yc_t):#把异常迹的索引从索引集合中剔除
    ind_t=[i for i in range(jz_size)]
    ind_t2 = [i for i in range(jz_size)]
    for j in ind_t:
        if j in yc_t:
            ind_t2.remove(j)

    new_ind=[]
    for ti in ind_t2:#把索引由[1,2,3]转为[[1],[2],[3]]
        t_list=ti
        new_ind.append(t_list)

    return new_ind

# sim=[[0,9,2,4,7],
#     [9,0,3,4,0],
#     [2,3,0,1,4],
#     [4,4,1,0,6],
#     [7,0,4,6,0]]
#
# k=2
# cu=[0,1,2,3,4]
# clusters=my_k(sim,k,cu)
# print(clusters)

import pickle

preprocessed_data_name = '../precess/sim_jz_act.pkl'#BOA
with open(preprocessed_data_name, 'rb') as f:
    sim_jz_act = pickle.load(f)

preprocessed_data_name = '../precess/Lev_sim.pkl'#Lev
with open(preprocessed_data_name, 'rb') as f:
    sim_jz_Lev = pickle.load(f)

preprocessed_data_name = '../precess/sim_con_cos2_yc.pkl'#后继关系控制流编码，余弦相似度度量，考虑了异常的存在
with open(preprocessed_data_name, 'rb') as f:
    sim_con_cos2_yc = pickle.load(f)

preprocessed_data_name = '../precess/sim_jz_time.pkl'#关联活动时间编码，高斯核函数度量
with open(preprocessed_data_name, 'rb') as f:
    sim_jz_time = pickle.load(f)

preprocessed_data_name = '../precess/yc_trace.pkl'#关联活动时间编码，高斯核函数度量
with open(preprocessed_data_name, 'rb') as f:
    yc_trace = pickle.load(f)

# sim=0.9*sim_con_cos2_yc+0.1*sim_jz_time
sim=sim_con_cos2_yc
# sim=sim_jz_Lev
sim_size=len(sim)
# yc_t=[4,24,41,93]
# yc_t=[4, 87, 166, 224, 276, 622, 896, 943, 1081, 1118, 1139, 1189, 1235, 1301, 1439, 1573, 1586, 1626, 1750, 1835, 1844, 1919, 1949, 1969, 1973, 2001, 2015, 2018, 2026, 2075]
# yc_t=[4, 87, 166, 224, 276, 622, 896, 943, 1081, 1118, 1139, 1189, 1235, 1301, 1439, 1573, 1586, 1626, 1750, 1835, 1844, 1919, 1949, 1969, 1973, 2001, 2015, 2018, 2026, 2075]
# yc_t=yc_trace
yc_t=[]
# print(yc_t)
clear_t=tichu_cu(sim_size,yc_t)
k=4
# clusters=my_k(sim,k,clear_t)
# print('未考虑频率的聚类结果：',clusters)





import pandas as pd

data = pd.read_excel('../precess/Log_data/PrepaidTravelCost.xlsx')
# data = pd.read_excel('../precess/Log_data/BPI_Challenge_2013_open_problems.xlsx')
dg = data.groupby(['case_id'])

def convert2str(x):
    x = str(x)
    return x
data['act']=data['act'].apply(convert2str)

def Stat_v(dgroup):#用字典的形式统计迹属于哪个变体，其中key是字符串迹
    v_dic={}
    dg_size=len(dgroup)
    ind=0
    for cid,df in dgroup:
        acts="".join(df.iloc[:,1])
        keys=[k for k in v_dic.keys()]
        if acts not in keys:
            val=[ind]
            v_dic[acts]=val
        else:
            v_dic[acts].append(ind)
        ind=ind+1
    return v_dic

v_dic=Stat_v(dg)
# print('变体集合:',v_dic)
# print(len(v_dic))

def change_key(dic,yc_t):#把key由字符串迹改为第一个变体的索引,并把异常迹的索引去掉
    new_v_dic={}
    for val in dic.values():
        k=val[0]
        new_v_dic[k]=val

    for t in yc_t:
        for val2 in new_v_dic.values():
            if t in val2:
                # print('剔除迹',t)
                val2.remove(t)
                break

    return new_v_dic

def adjust_cluster(cu,new_dic):
    flg=False#设立一个flag，如果为False则没进行调整
    for val in new_dic.values():
        ad_dic={}
        for c_ind,c in zip(range(len(cu)),cu):
            count=[i for i in val if i in c]
            if len(count)==0:#表示变体全部不在当前类别
                continue
            if len(count)==len(val):#表示变体全部在一个类别
                break
            ad_dic[c_ind]=count

        if len(ad_dic)==0:#如果字典为空则没进行调整，否则进行了调整
            continue
        flg=True
        temp_size=0#表示属于同一个变体的迹所在同一个类别最多的数量
        max_lb_ind=0#表示属于同一个变体的迹所在同一个类别最多的数量的类别索引
        for k,v in ad_dic.items():
            if len(v)>temp_size:
                temp_size=len(v)
                max_lb_ind=k

        for k2,v2 in ad_dic.items():#对类别进行调整
            if k2==max_lb_ind:
                continue
            for t_ind in v2:
                cu[max_lb_ind].append(t_ind)
                cu[k2].remove(t_ind)

    if flg:
        print('分类已经根据频率进行了调整！！！')
    else:
        print('属于同一个变体的迹全部分类到一个类别中，无须进行调整！！！')
    # print(cu)
    return cu


new_v_dic=change_key(v_dic,yc_t)
# print('new变体集合:',new_v_dic)
# print(len(new_v_dic))
import copy


# print('考虑了频率的聚类结果：',clusters_fre)



def neiju(sim_jz, clusters):
    re = []
    for i, clu in zip(range(len(clusters)), clusters):
        temp_re = []
        temp_c = 0
        temp_he = 0
        for c in clu:
            neijuhe = 0
            for p in clu:
                if c == p:
                    continue
                neijuhe = neijuhe + sim_jz[c][p]

            if temp_he < neijuhe:
                temp_he = neijuhe
                temp_c = c

        temp_re.append(temp_c)
        temp_re.append(temp_he)
        temp_re.append(len(clusters[i]))
        re.append(temp_re)
    return re



def cal_neiju(elv):
    sum=0
    elv_size=len(elv)
    for cl in elv:
        cl_sum=cl[1]/cl[2]
        sum=sum+cl_sum
    re=sum/elv_size
    return re


def ouhe(sim,elv):
    elv_size=len(elv)
    ouhe_sum=0
    for i in range(elv_size-1):
        for j in range(i+1,elv_size):
             ouhe_sum=ouhe_sum+sim[elv[i][2]][elv[j][2]]
    # print("耦合：", ouhe_sum)
    l=elv_size*(elv_size-1)/2
    re=ouhe_sum/l
    return re

temp_nj=0
temp_ouhe=0
temp_cha=0
temp_clusters=0
temp_clusters_fre=0
for i in range(3):
    clusters = my_k(sim, k, clear_t)
    # print('未考虑频率的聚类结果：', clusters)


    # clusters2 = copy.deepcopy(clusters)
    # clusters_fre = adjust_cluster(clusters2, new_v_dic)

    elv=neiju(sim,clusters)
    print(elv)
    cal_nj=cal_neiju(elv)
    cal_ouhe = ouhe(sim, elv)
    cha=cal_nj-cal_ouhe
    if cha>temp_cha:
        temp_nj=cal_nj
        temp_ouhe=cal_ouhe
        temp_cha=cha
        temp_clusters=clusters
        preprocessed_data_name = 'kmean_cluster_no_fre.pkl'
        with open(preprocessed_data_name, 'wb') as f:
            pickle.dump(clusters, f, protocol=2)

        print('写入成功no_fre！！！')
        # temp_clusters_fre=clusters_fre
        # preprocessed_data_name = 'kmean_cluster_with_fre.pkl'
        # with open(preprocessed_data_name, 'wb') as f:
        #     pickle.dump(clusters_fre, f, protocol=2)
        # print('写入成功with_fre！！！')

        print("内聚：",cal_nj)

        print("耦合：",cal_ouhe)

print("内聚：",temp_nj)
print("耦合：",temp_ouhe)
for i in range(k):
    print(len(temp_clusters[i]))
# print('------------------')
# for j in range(k):
#     print(len(temp_clusters_fre[j]))

#
# print('--------------')
#
# for i in range(k):
#     print(len(clusters_fre[i]))
#
# preprocessed_data_name =  'kmean_cluster_with_fre.pkl'
# with open(preprocessed_data_name, 'wb') as f:
#     pickle.dump(clusters_fre, f, protocol=2)
#
# print('写入成功with_fre！！！')