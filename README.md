# control-time-cluster
### con_sim_cos_2.py 活动行为关系的控制流编码及相似性计算  
time_part_yc.py 提取各个活动的关联活动对  
Time_sim_jz 提取各个活动的最近关联活动对，计算关联时间差，标准化，计算时间属性的相似度  
my_kmeans.py 聚合控制流和时间视角的集成相似度矩阵，使用k中心聚类，基于属于同一个变体的迹的频率调整聚类结果  
cal_fitness.py 使用Pm4py库，基于启发式挖掘出的Petri网，计算fitness和Precision  
Activity_part.py BOA编码及相似性计算  
Lev_sim.py 迹的原编码并且采用Levenshtein距离度量  
登录系统文件 保存了登录系统的各个子流程的Petri网文件  
