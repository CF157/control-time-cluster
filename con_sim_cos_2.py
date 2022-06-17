import pandas as pd
import numpy as np

#控制流：后继编码，余弦相似度度量，考虑了异常的存在
# data=pd.read_excel('../sc/m01log01.xlsx')
# data=pd.read_excel('v02.xlsx')
data=pd.read_excel('../precess/Log_data/PrepaidTravelCost.xlsx')
# data=pd.read_excel('../precess/Log_data/Log_v/promlog_v.xlsx')
dg=data.groupby(['case_id'])
thre=0.3#判断是否是并发行为的阈值
v=[]
# case_id=[]
for id,d in dg:
    # print(id)
    a=list(d.iloc[:,1])
    v.append(a)
    # case_id.append(id)
print(len(v))
# print(case_id)
# print(len(case_id))

def tiqubianti(case_id,group):#提取数据集中的迹变体
    re=[]
    for i,gp in group:
        if i in case_id:
            re.append(gp)
    af=pd.concat(re)
    return af

def  xieru(df):#把迹变体写入excel
        df.to_excel('v01.xlsx',index=False)
        print('变体写入成功！！！')
# af=tiqubianti(case_id,dg)
# xieru(af)
def bianma(v):#对变体中的迹进行后继编码,找出子序列
    seq_dic={}
    for i in v:
        l=len(i)
        for n in range(l-1):
            s1=i[n]
            s2=i[n+1]
            subseq=str(s1)+str(s2)
            key=[k for k in seq_dic.keys()]
            if subseq in key:
                seq_dic[subseq]=seq_dic[subseq]+1
            else:
                seq_dic[subseq]=1
    return seq_dic
seq_dic=bianma(v)
# print('seq_dic',seq_dic)

def p_s(seq_zd,yz):#找出并发序列,不包括自循环
    p_seq=[]
    seq=[k for k in seq_zd.keys()]
    for i,p in zip(range(len(seq)),seq):
        if i+1<len(seq):
            for j in range(i+1,len(seq)):
                r=seq[j]
                if r[0]==p[1] and r[1]==p[0]:
                    val1=seq_zd[p]
                    val2=seq_zd[r]
                    val_max=max(val1,val2)
                    fenzi=abs(val1-val2)
                    prb=fenzi/val_max
                    if prb<yz:
                        p_seq.append(r)
                        break
                    else:break
    return p_seq
p_seq=p_s(seq_dic,thre)
print('并发序列：',p_seq)
seq=[k for k in seq_dic.keys()]
def tichu(p_seq2,seq2):#删除变体编码中的并发行为
    for i in p_seq2:
        if i in seq2:
            seq2.remove(i)
    return seq2
aseq=tichu(p_seq,seq)#最终的行为类别
print('最终的行为类别',aseq)
print(len(aseq))
def encodev(v):#对变体中的迹进行后继编码,可以改为对日志的所有迹（不一定只针对变体）
    e_t=[]
    for i in v:
        l = len(i)
        t=[]
        for n in range(l - 1):
            s1 = i[n]
            s2 = i[n + 1]
            subseq = str(s1) + str(s2)
            t.append(subseq)
        e_t.append(t)
    return e_t
e_t=encodev(v)
# print(len(e_t))
# print(e_t[0])
def fanx(p_seq):
    r_p_s=[]
    for p in p_seq:
        st=str(p[1])+(p[0])
        r_p_s.append(st)
    return r_p_s

# r_p_s=fanx(p_seq)
# print('r_p_s',r_p_s)
def jijuzhen(seq,e_t):
    jz=[]
    for t in e_t:
        tem = [0 for i in range(len(seq))]
        for ts in t:
             if ts in seq:
                suoyin=seq.index(ts)
                tem[suoyin]=tem[suoyin]+1
             else:
                 fs=str(ts[1])+str(ts[0])
                 suoyin = seq.index(fs)
                 tem[suoyin] = tem[suoyin] + 1
        jz.append(tem)
    return jz

print('正在对迹编码……')
jz=jijuzhen(aseq,e_t)
print('迹编码结束！！')
# print(e_t[4])
# print(jz[4])
# print(len(aseq))
# print(len(jz[4]))

def cal_sim2(jz):
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
                fenzi=np.sum(ti*tj)
                fenmu=pow(np.sum(ti**2),0.5)*pow(np.sum(tj**2),0.5)
                if fenzi==0:
                    sim_jz[i][j] =0.00001
                else:
                    sim_jz[i][j]=fenzi/fenmu
    return sim_jz

print('正在计算相似度……')
sim_jz_cos=cal_sim(jz)
print(len(sim_jz_cos))
print(sim_jz_cos[:7])
print('结束！')




import pickle


preprocessed_data_name =  'sim_con_cos2_yc.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(sim_jz_cos, f, protocol=2)

print('写入成功！！！')

preprocessed_data_name = 'bf_seq_yc.pkl'
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(p_seq, f, protocol=2)

print('并发序列，写入成功！！！')




