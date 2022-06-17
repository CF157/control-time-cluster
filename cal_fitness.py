import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import os

# log = xes_importer.apply(os.path.join("Log","log6.xes"))
log = xes_importer.apply(os.path.join("../precess/xes_bag","bpi20_cl_no_time_kmean0.xes"))
# log = xes_importer.apply(os.path.join("../precess/xes_bag","Sepsis_Cases.xes"))
# log = xes_importer.apply(os.path.join("../precess/xes_bag","BPI_Challenge_2013_open_problems.xes"))
# log = xes_importer.apply(os.path.join("../precess/xes_bag/Act_bpi20","Sepsis_cl_my_kmean0.xes"))
# log = xes_importer.apply(os.path.join("../precess/xes_bag","Sepsis_cl_my_kmean3.xes"))

# from pm4py.algo.discovery.inductive import algorithm as inductive_miner
# net,im,fm=inductive_miner.apply(log)

# from pm4py.algo.discovery.alpha import algorithm as alpha_miner
# net, im, fm = alpha_miner.apply(log)

# from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
# net, im, fm = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99,
#                                                     heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT:1,
#                                                     heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES:1,
#                                                     heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH:2})

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
net, im, fm = heuristics_miner.apply(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})

# from pm4py.visualization.petrinet import visualizer as pn_visualizer
# gviz = pn_visualizer.apply(net, im, fm)
# pn_visualizer.view(gviz)
#
# from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
# pnml_exporter.apply(net, im, "petri.pnml",fm)


# from pm4py.evaluation.replay_fitness import evaluator as replay_fitness
# log_fitness=replay_fitness.apply(log,net,im,fm)
# print(log_fitness)
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
fitness = replay_fitness_evaluator.apply(log, net, im, fm)
print('fitness:',fitness)



# from pm4py.algo.evaluation.precision import evaluator
# log_precision=evaluator.apply(log,net,im,fm)
# print(log_precision)
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
prec = precision_evaluator.apply(log, net, im, fm)
print('precision',prec)

# from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
# gen = generalization_evaluator.apply(log, net, im, fm)
# print('g',gen)

# from pm4py.algo.conformance.alignments import algorithm as alignments
# alignments = alignments.apply_log(log, net, im, fm)
# print("alignments success")
# from pm4py.evaluation.replay_fitness import evaluator as replay_fitness
# log_fitness = replay_fitness.evaluate(alignments, variant=replay_fitness.Variants.ALIGNMENT_BASED)
# print('log2',log_fitness)