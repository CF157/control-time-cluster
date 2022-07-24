import pickle
import pandas as pd
import numpy as np

preprocessed_data_name1 = 'time_dic_yc.pkl'
with open(preprocessed_data_name1, 'rb') as f:
    time_dic = pickle.load(f)

preprocessed_data_name2 = 'bf_seq_yc.pkl'
with open(preprocessed_data_name2, 'rb') as f:
    B_F=pickle.load(f)

# B_F=['EK', 'FE', 'TU', 'FK', 'QP','GE','GK']#并发序列，不包括自循环
def pd_bf(val,bf):#找到关联活动的并发活动
    new_val=[]
    s0=val[0]
    new_val.append(s0)
    for s in val:
        st1=s0+s
        st2 = s + s0
        if st1 in bf or st2 in bf:
            new_val.append(s)
    return new_val


def g_l_time(dic,bf):#找出每个活动的直接依赖活动
    G_L={}
    for key,val in dic.items():
        if len(val)<1:
            G_L[key]=[]
            continue
        value=pd_bf(val,bf)
        G_L[key]=value
    return G_L

print(time_dic)
new_time_dic=g_l_time(time_dic,B_F)
print('new_time_dic',new_time_dic)

data = pd.read_excel('../precess/Log_data/PrepaidTravelCost.xlsx')#整个日志的迹
dg = data.groupby(['case_id'])

data['date'] = pd.to_datetime(data['date'], errors='coerce')


def convert2str(x):
    x = str(x)
    return x
data['act']=data['act'].apply(convert2str)

# def calculateCumDuration(df):
#     df['Duration'] = (df['date'] - df['date'].iloc[0])
#     return df

def calculateCumDuration(df):
    l=len(df)
    T=[]
    for i in range(l):
        t=(df['date'].iloc[i] - df['date'].iloc[0])
        T.append(t)
    df['Duration']=T
    return df

def convert2seconds(x):
    x = x.total_seconds()
    return x


duration_df = pd.DataFrame(pd.DataFrame(columns=list(data) + ['Duration']))
for case, group in dg:
    group = calculateCumDuration(group)
    group['Duration'] = group['Duration'].apply(convert2seconds)
    duration_df = duration_df.append(group)

df_gp = duration_df.groupby(['case_id'])
#
# preprocessed_data_name =  'dur_time_df.pkl'
# with open(preprocessed_data_name, 'wb') as f:
#     pickle.dump(duration_df, f, protocol=2)
#
# print('dur_time写入成功！！！')

# print(duration_df.iloc[20:40,:])

# preprocessed_data_name = '../precess/dur_mean.pkl'
# with open(preprocessed_data_name, 'rb') as f:
#     df_time_mean = pickle.load(f)
#
# df_gp=df_time_mean.groupby(['case_id'])

def guan_lian_act(dit):  #把关联活动变成列表['CA', 'KC', 'EC', 'FC', 'IF', 'GF', 'DG', 'DE',...]
    g_l_act_list=[]
    for key,val in dit.items():
        if len(val)<1:
            continue
        for a in val:
            str1=str(key)+str(a)
            g_l_act_list.append(str1)
    return g_l_act_list

g_l_a_list=guan_lian_act(new_time_dic)
print('g_l_a_list',g_l_a_list)

def trace_list(df_group):  # 把每个迹变成活动列表['A', 'C', 'K', 'E', 'F', 'I', 'G', 'D', 'B']
    traces = []
    for i, g in df_group:
        trace = list(g.iloc[:, 1])
        traces.append(trace)
    return traces

traces=trace_list(df_gp)

def t_a(t):#统计每个迹中包含的活动种类
    act=[]
    for a in t:
        if a not in act:
            # str1=str(a)
            act.append(a)
    return act

def f(dur_g,g_a_l,dit):#统计每个迹中所包含的关联活动所持续的时间，输入为dur_g、关联活动列表、关联活动字典
    t_time=[]
    for caseid,df in dur_g:
        t=list(df.iloc[:, 1])
        t_act=t_a(t)
        temp_time = [0 for i in range(len(g_a_l))]
        for ta in t_act:
            val=dit[ta]
            if len(val)<1:#判断该活动是否有前置活动
                continue
            for v in val:
                ind_aft=t.index(ta)
                t2=t[:ind_aft]
                if v not in t2:#判断在ta之前是否有活动v（可能会有异常迹，缺失或者错位，所关联的活动）
                    continue
                ind_pre = t.index(v)
                             
                time_list=list(df.iloc[:,3])
                time1=time_list[ind_pre]
                time2=time_list[ind_aft]
                dur_time=(time2-time1)/3600
                # yuzhi_dur=
                # if dur_time>
                str1=ta+v
                ind_glb=g_a_l.index(str1)
                temp_time[ind_glb]=dur_time
        t_time.append(temp_time)
    return t_time
ts_time_list=f(df_gp,g_l_a_list,new_time_dic)
# print(g_l_a_list)
# print(len(g_l_a_list))
print('ts_time_list',ts_time_list[:5])
preprocessed_data_name =  'time_list.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(ts_time_list, f, protocol=2)

print('time_list写入成功！！！')
# print(len(ts_time_list))
# print(len(ts_time_list[0]))
# print(len(traces))

def normal_bz2(time_list):#横向标准化，会出现过大值占主导地位
    t_arry=np.array(time_list)
    result=[]
    ind=0
    for arr in t_arry:
        pf = arr * arr
        he = np.sum(pf)
        gh = np.sqrt(he)
        if gh==0:
            print('ind',ind)
            # print('yyyy',time_list[ind])
            result.append(time_list[ind])
            ind = ind + 1
            continue
        re = list(arr / gh)
        result.append(re)
        ind=ind+1
    return result

def normal_bz(time_list):#纵向标准化z-score
    t_arry = np.array(time_list)
    result = []
    mean_list=[]
    std_list=[]
    arr_size=len(t_arry[0])
    # print('arr_size',arr_size)
    for i in range(arr_size):
        temp_col=t_arry[:,i]
        temp_no_zero=[p for p in temp_col if p!=0]
        if i==8:
            # temp_no_zero = [1]
            print('temp_no_zero',temp_no_zero)
        col_mean = np.mean(temp_no_zero)
        col_std = np.std(temp_no_zero)
        t_arry[:, i] = [j if j != 0 else col_mean for j in temp_col]
        mean_list.append(col_mean)
        std_list.append(col_std)
    # print('new_time_list',t_arry[:5])

    # print('mean_list',mean_list)
    arr_mean=np.array(mean_list)
    arr_std=np.array(std_list)
    print('std',arr_std)
    for arr in t_arry:
        re=list((arr-arr_mean)/(arr_std+0.999))
        result.append(re)
    # print('result',len(result[0]))
    return result

def normal_bz3(time_list):#纵向标准化max-min标准化
    t_arry = np.array(time_list)
    result = []
    min_list=[]
    max_list=[]
    arr_size=len(t_arry[0])
    # print('arr_size',arr_size)
    for i in range(arr_size):
        temp_col=t_arry[:,i]
        col_min = np.min(temp_col)
        col_max = np.max(temp_col)
        min_list.append(col_min)
        max_list.append(col_max)
    # print('new_time_list',t_arry[:5])

    print('min_list',min_list)
    arr_min=np.array(min_list)
    arr_max=np.array(max_list)
    for arr in t_arry:
        re=list((arr-arr_min)/(arr_max-arr_min))
        result.append(re)
    # print('result',len(result[0]))
    return result

time_normal=normal_bz(ts_time_list)
print('time_normal',time_normal[0])
print('time_normal',time_normal[5])
preprocessed_data_name =  'time_normal.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(time_normal, f, protocol=2)

print('写入成功！！！')

def cal_time_sim2(t_norm):#高斯核函数
    l = len(t_norm)
    sim_jz = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i==j:
                continue
            else:
                num1=np.array(t_norm[i])
                num2=np.array(t_norm[j])
                zc=num1-num2
                pf=zc*zc
                he=np.sum(pf)
                zs=-1*he/2
                sim_jz[i][j]=np.exp(zs)
    return sim_jz

def cal_time_sim(t_norm):#活动包和时间的高斯核函数
    l = len(t_norm)
    sim_jz = np.zeros((l, l))
    sqr1_jz=np.zeros((l, l))
    sqr2_jz = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i==j:
                continue
            else:
                t1 = [1 if n != 0 else 0 for n in t_norm[i]]
                ti = np.array(t1)
                t2 = [1 if m != 0 else 0 for m in t_norm[j]]
                tj = np.array(t2)
                ti_len=[1 for li in ti if li==1]
                tj_len=[1 for lj in tj if lj==1]
                ti_tj=[1 for i_j in range(len(t1)) if ti[i_j]==1 and tj[i_j]==1]
                # zc1=(ti-tj)
                # pf1=zc1*zc1
                # sqr1 = pow(np.sum(pf1), 0.5)
                # sqr1_jz[i][j]=sqr1
                zong_len=(len(ti_len)+len(tj_len))
                if zong_len==0:
                    sqr1_jz[i][j] = 0
                else:
                    sqr1_jz[i][j]=2*len(ti_tj)/zong_len
                jiao=ti*tj
                num1=np.array(t_norm[i])
                num2=np.array(t_norm[j])
                zc=num1-num2
                pf=zc*zc*jiao
                # pf=zc*zc
                gh=pow(np.sum(pf), 0.5)
                sqr2=gh
                sqr2_jz[i][j]=sqr2
    sqr1_arr=np.array(sqr1_jz)
    # sqr1_sim=1-sqr1_arr/np.max(sqr1_arr)
    sqr1_sim=sqr1_arr
    e=-0.1
    sqr2_arr = np.array(sqr2_jz)/e
    # sqr2_sim = 1 - sqr2_arr / np.max(sqr2_arr)
    sqr2_sim=np.exp(sqr2_arr)
    print('max',np.max(sqr1_arr),np.max(sqr2_arr))
    # sim_jz = 0.5 * sqr2_sim + 0.5 * sqr1_sim
    sim_jz = 0.5*sqr1_sim + 0.5*sqr2_sim
    # sim_jz = sqr2_arr
    for ii in range(l):
        for jj in range(l):
            if ii==jj:
                sim_jz[ii][jj]=0


    return sim_jz

def cal_time_sim3(jz):#余弦相似度与高斯核函数
    l=len(jz)
    sim_jz = np.zeros((l, l))
    for i in range(len(jz)):
        for j in range(len(jz)):
            if i==j:
                continue
            else:
                t1=[1 if n!=0 else 0 for n in jz[i] ]
                ti=np.array(t1)
                t2=[1 if m!=0 else 0 for m in jz[j] ]
                tj=np.array(t2)
                fenzi=np.sum(ti*tj)
                fenmu=pow(np.sum(ti**2),0.5)*pow(np.sum(tj**2),0.5)
                # re1=fenzi/fenmu
                if fenzi==0:
                    re1 =0.00001
                else:
                    re1=fenzi/fenmu#余弦相似度

                num1 = np.array(jz[i])
                num2 = np.array(jz[j])
                zc = num1 - num2
                pf = zc * zc
                he = np.sum(pf)
                zs = -1 * he / 2
                re2=np.exp(zs)#高斯核函数
                # sim_jz[i][j] = 0.5*re1+0.5*re2
                sim_jz[i][j] = re2

    return sim_jz

time_sim_jz=cal_time_sim(time_normal)
print(time_sim_jz[:7])
print(len(time_sim_jz))
#
preprocessed_data_name =  'sim_jz_time.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(time_sim_jz, f, protocol=2)

print('写入成功！！！')
