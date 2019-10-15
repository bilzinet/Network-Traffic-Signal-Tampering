# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:10:16 2019

@author: btt1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import pickle

cwd = os.getcwd()
os.chdir(cwd)

def pareto_normalized(k):

    par_front = np.load('Expt-no-{}/Optimal_Pareto_frontier_exptno_{}.npy'.format(k,k))
    par_front.sort(axis=0)
    par_front_max = np.max(par_front, axis=0)
    par_front_min = np.min(par_front, axis=0)
    par_front_norm = (par_front-par_front_min)/(par_front_max - par_front_min)
    
    return par_front, par_front_norm

def surrogate_pareto_normalized(k):

    with open('Expt-no-{}/Adversarial_assignment_results_exptno_{}.npy'.format(k,k), 'rb') as file:
        res = pickle.load(file)
    objs = np.array(list(res['objs'].values()))
    
    objs = np.abs(objs)
    objs.sort(axis=0)
    objs_max = np.max(objs, axis=0)
    objs_min = np.min(objs, axis=0)
    objs_norm = (objs-objs_min)/(objs_max - objs_min)
    
    return objs, objs_norm

# -------------------- Analysis of objective functions ---------------------- #

col_width=3
fig_width=col_width
fig_height=col_width
lab_font=13
tick_font=12
leg_fond=12


expt = 1
fig = plt.figure(figsize=(fig_width,fig_height))
par_front, par_front_norm = surrogate_pareto_normalized(expt)
plt.scatter(par_front[:,1], par_front[:,0]/1000, s=3.5)
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
plt.grid()
plt.tight_layout()
plt.savefig('Pareto-front-lowuniform-meddemand.pdf',bbox_inches = 'tight',pad_inches = 0.1)

expt = 11
fig = plt.figure(figsize=(fig_height,fig_height))
par_front, par_front_norm = surrogate_pareto_normalized(expt)
plt.scatter(par_front[:,1], par_front[:,0]/1000, s=3.5)
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
plt.grid()
plt.tight_layout()
plt.savefig('Pareto-front-highuniform-meddemand.pdf',bbox_inches = 'tight',pad_inches = 0.1)




# --------------------- Impact of Network Structure ------------------------- #

col_width=3
fig_width=col_width
fig_height=col_width
lab_font=13
tick_font=12
leg_fond=12

'''
Medium Demand:
    
Expt 2: 450 timestep, 200 demand, Low-uniform
Expt 7: 450 timestep, 200 demand, Med-uniform
Expt 12: 450 timestep, 200 demand, High-uniform
Expt 17: 450 timestep, 200 demand, High-random

'''
#expts = np.array([2, 7, 12, 17])
#labels = ['Low-Uniform', 'Med-Uniform', 'High-Uniform', 'High-Random']
#fig = plt.figure(figsize=(7,7))
#plt.title("Impact of different network structure", fontsize=14)
#for i, expt in enumerate(expts):
#    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=2)
#plt.xlabel("Number of Signal changes", fontsize=12); 
#plt.ylabel("Difference in the number of cumulative flows", fontsize=12)
#plt.legend()
#plt.grid()
#plt.savefig('Net-struct-comp-meddemand.pdf')

expts = np.array([2, 7, 12, 17])
labels = ['Low-Uniform', 'Med-Uniform', 'High-Uniform', 'High-Random']
colors = ['C0', 'C1', 'C2', 'C3']
fig = plt.figure(figsize=(fig_width, fig_height))
#plt.title("Traffic=200 vehs, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of different network structure (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=2, facecolors=colors[i], edgecolors=colors[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Network traffic impact ($z_1$)", fontsize=lab_font)
plt.ylim([0,1]); plt.xlim([0,1])
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=leg_fond)
plt.grid()
plt.tight_layout()
plt.savefig('Net-struct-comp-meddemand-norm.pdf')


'''
Low Demand:
    
Expt 4: 450 timestep, 100 demand, Low-uniform
Expt 9: 450 timestep, 100 demand, Med-uniform
Expt 14: 450 timestep, 100 demand, High-uniform
Expt 19: 450 timestep, 100 demand, High-random

'''
expts = np.array([4, 9, 14, 19])
labels = ['Low-Uniform', 'Med-Uniform', 'High-Uniform', 'High-Random']
colors = ['C0', 'C1', 'C2', 'C3']
fig = plt.figure(figsize=(fig_width, fig_height))
#plt.title("Traffic=200 vehs, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of different network structure (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=2, facecolors=colors[i], edgecolors=colors[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font, labelpad=0.5)
plt.ylabel(r"Network traffic impact ($z_1$)", fontsize=lab_font, labelpad=0.5)
plt.ylim([0,1]); plt.xlim([0,1])
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=leg_fond)
plt.grid()
plt.tight_layout()

plt.savefig('Net-struct-comp-lowdemand-norm.pdf')


'''
High Demand:
    
Expt 5: 450 timestep, 300 demand, Low-uniform
Expt 10: 450 timestep, 300 demand, Med-uniform
Expt 15: 450 timestep, 300 demand, High-uniform
Expt 20: 450 timestep, 300 demand, High-random

'''
expts = np.array([5, 10, 15, 20])
labels = ['Low-Uniform', 'Med-Uniform', 'High-Uniform', 'High-Random']
colors = ['C0', 'C1', 'C2', 'C3']
fig = plt.figure(figsize=(fig_width, fig_height))
#plt.title("Traffic=200 vehs, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of different network structure (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=2, facecolors=colors[i], edgecolors=colors[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font, labelpad=0.5)
plt.ylabel(r"Network traffic impact ($z_1$)", fontsize=lab_font, labelpad=0.5)
plt.ylim([0,1]); plt.xlim([0,1])
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=leg_fond)
plt.grid()
plt.tight_layout()

plt.savefig('Net-struct-comp-highdemand-norm.pdf')


# --------------------- Impact of Network Demand ---------------------------- #

col_width=3
fig_width=col_width
fig_height=col_width
lab_font=14
tick_font=13
leg_fond=12


'''
Low Uniform Network:
    
Expt 2: 450 timestep, 200 demand, Med-demand
Expt 4: 450 timestep, 100 demand, Low-demand
Expt 5: 450 timestep, 300 demand, High-demand

'''
expts = np.array([5, 2, 4])
labels = ['High traffic', 'Medum traffic', 'Low traffic']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Demand-comp-lowuniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)

'''
Medium Uniform Network:
    
Expt 7: 450 timestep, 200 demand, Med-demand
Expt 9: 450 timestep, 100 demand, Low-demand
Expt 10: 450 timestep, 300 demand, High-demand

'''
expts = np.array([10, 7, 9])
labels = ['High traffic', 'Medum traffic', 'Low traffic']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Demand-comp-meduniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)

'''
High Uniform Network:
    
Expt 12: 450 timestep, 200 demand, Med-demand
Expt 14: 450 timestep, 100 demand, Low-demand
Expt 15: 450 timestep, 300 demand, High-demand

'''
expts = np.array([15, 12, 14])
labels = ['High traffic', 'Medum traffic', 'Low traffic']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Demand-comp-highuniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)


'''
High Random Network:
    
Expt 17: 450 timestep, 200 demand, Med-demand
Expt 19: 450 timestep, 100 demand, Low-demand
Expt 20: 450 timestep, 300 demand, High-demand

'''
expts = np.array([20, 17, 19])
labels = ['High traffic', 'Medum traffic', 'Low traffic']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Demand-comp-highrandom.pdf',bbox_inches = 'tight',pad_inches = 0.1)


# --------------------------------------------------------------------------- #

# ----------------- Impact of Time period of attack ------------------------- #

'''
Low Uniform Network:
    
Expt 1: 300 timestep, 200 demand, Med-demand
Expt 2: 450 timestep, 200 demand, Med-demand
Expt 3: 600 timestep, 200 demand, Med-demand

'''
expts = np.array([1, 2, 3])
labels = ['10 mins', '15 mins', '20 mins']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
#    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=20, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Time-comp-lowuniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)

'''
Medium Uniform Network:
    
Expt 7: 450 timestep, 200 demand, Med-demand
Expt 9: 450 timestep, 100 demand, Low-demand
Expt 10: 450 timestep, 300 demand, High-demand

'''
expts = np.array([6, 7, 8])
labels = ['10 mins', '15 mins', '20 mins']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
#    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=20, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Time-comp-meduniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)

'''
High Uniform Network:
    
Expt 12: 450 timestep, 200 demand, Med-demand
Expt 14: 450 timestep, 100 demand, Low-demand
Expt 15: 450 timestep, 300 demand, High-demand

'''
expts = np.array([11, 12, 13])
labels = ['10 mins', '15 mins','20 mins']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
#    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=20, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Time-comp-highuniform.pdf',bbox_inches = 'tight',pad_inches = 0.1)


'''
High Random Network:
    
Expt 17: 450 timestep, 200 demand, Med-demand
Expt 19: 450 timestep, 100 demand, Low-demand
Expt 20: 450 timestep, 300 demand, High-demand

'''
expts = np.array([16, 17, 18])
labels = ['10 mins', '15 mins', '20 mins']
colors = ['C0', 'C1', 'C2']
alphas = [1.0, 1.0, 1.0]
fig = plt.figure(figsize=(fig_width,fig_height))
#plt.title("Low uniform network, Timesteps=450 steps", fontsize=12)
#plt.suptitle("Impact of various demand levels (Normalized)", fontsize=14)
for i, expt in enumerate(expts):
    par_front, par_front_norm = surrogate_pareto_normalized(expt)
#    plt.plot(par_front[:,1], par_front[:,0], label=labels[i], linewidth=1)
#    plt.scatter(par_front_norm[:,1], par_front_norm[:,0], label=labels[i], s=20, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
    plt.scatter(par_front[:,1], par_front[:,0]/1000, label=labels[i], s=1, facecolors=colors[i], edgecolors=colors[i], alpha=alphas[i])
plt.xlabel(r"Noticeability measure ($z_2$)", fontsize=lab_font)
plt.ylabel(r"Adversarial impact ($\times 10^{3} z_1$)", fontsize=lab_font)
plt.xticks(fontsize=tick_font); plt.yticks(fontsize=tick_font)
#plt.legend(fontsize=9)
plt.grid()
plt.tight_layout()
plt.savefig('Time-comp-highrandom.pdf',bbox_inches = 'tight',pad_inches = 0.1)

# --------------------------------------------------------------------------- #


par_front_l, par_front_norm = surrogate_pareto_normalized(19)
par_front_m, par_front_norm = surrogate_pareto_normalized(17)
par_front_h, par_front_norm = surrogate_pareto_normalized(20)

slopes_l = []
for i in range(par_front_l.shape[0]-1):
    y1, x1 = par_front_l[i,:]
    y2, x2 = par_front_l[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_l.append(m)
slopes_m = []
for i in range(par_front_m.shape[0]-1):
    y1, x1 = par_front_m[i,:]
    y2, x2 = par_front_m[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_m.append(m)
slopes_h = []
for i in range(par_front_h.shape[0]-1):
    y1, x1 = par_front_h[i,:]
    y2, x2 = par_front_h[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_h.append(m)

plt.figure(figsize=(7,6))
plt.scatter(np.arange(len(slopes_h)), slopes_h, s=20, facecolors='C0', edgecolors='C0', label='High demand')
plt.scatter(np.arange(len(slopes_m)), slopes_m, s=20, facecolors='C1', edgecolors='C1', label='Medium demand')
plt.scatter(np.arange(len(slopes_l)), slopes_l, s=20, facecolors='C2', edgecolors='C2', label='Low demand')
plt.xlabel("Number of data points", fontsize=18)
plt.ylabel("Slopes (= network vulnerability)", fontsize=18)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.savefig('Slope-comp-highrandom.pdf')

#plt.figure()
#n_bins = 20
#fig, axes = plt.subplots(nrows=1, ncols=1)
#axes.hist(slopes_l, n_bins, normed=1, histtype='step', stacked=False, fill=False)
#axes.hist(slopes_m, n_bins, normed=1, histtype='step', stacked=False, fill=False)
#axes.hist(slopes_h, n_bins, normed=1, histtype='step', stacked=False, fill=False)

#data = [slopes_h, slopes_m, slopes_l]
#fig, axes = plt.subplots()
#axes.boxplot(data, showfliers=False)

# -----------

par_front_l, par_front_norm = surrogate_pareto_normalized(14)
par_front_m, par_front_norm = surrogate_pareto_normalized(12)
par_front_h, par_front_norm = surrogate_pareto_normalized(15)

slopes_l = []
for i in range(par_front_l.shape[0]-1):
    y1, x1 = par_front_l[i,:]
    y2, x2 = par_front_l[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_l.append(m)
slopes_m = []
for i in range(par_front_m.shape[0]-1):
    y1, x1 = par_front_m[i,:]
    y2, x2 = par_front_m[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_m.append(m)
slopes_h = []
for i in range(par_front_h.shape[0]-1):
    y1, x1 = par_front_h[i,:]
    y2, x2 = par_front_h[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_h.append(m)
    
plt.figure(figsize=(5,5))
plt.scatter(np.arange(len(slopes_h)), slopes_h, s=20, facecolors='none', edgecolors='C0', label='High demand')
plt.scatter(np.arange(len(slopes_m)), slopes_m, s=20, facecolors='none', edgecolors='C1', label='Medium demand')
plt.scatter(np.arange(len(slopes_l)), slopes_l, s=20, facecolors='none', edgecolors='C2', label='Low demand')
plt.xlabel("Number of data points", fontsize=13)
plt.ylabel("Slope", fontsize=13)
plt.legend()
plt.grid()
plt.savefig('Slope-comp-highuniform.pdf')


par_front_l, par_front_norm = surrogate_pareto_normalized(9)
par_front_m, par_front_norm = surrogate_pareto_normalized(7)
par_front_h, par_front_norm = surrogate_pareto_normalized(10)

slopes_l = []
for i in range(par_front_l.shape[0]-1):
    y1, x1 = par_front_l[i,:]
    y2, x2 = par_front_l[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_l.append(m)
slopes_m = []
for i in range(par_front_m.shape[0]-1):
    y1, x1 = par_front_m[i,:]
    y2, x2 = par_front_m[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_m.append(m)
slopes_h = []
for i in range(par_front_h.shape[0]-1):
    y1, x1 = par_front_h[i,:]
    y2, x2 = par_front_h[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_h.append(m)

plt.figure(figsize=(5,5))
plt.scatter(np.arange(len(slopes_h)), slopes_h, s=20, facecolors='none', edgecolors='C0', label='High demand')
plt.scatter(np.arange(len(slopes_m)), slopes_m, s=20, facecolors='none', edgecolors='C1', label='Medium demand')
plt.scatter(np.arange(len(slopes_l)), slopes_l, s=20, facecolors='none', edgecolors='C2', label='Low demand')
plt.xlabel("Number of data points", fontsize=13)
plt.ylabel("Slope", fontsize=13)
plt.legend()
plt.grid()
plt.savefig('Slope-comp-meduniform.pdf')


par_front_l, par_front_norm = surrogate_pareto_normalized(4)
par_front_m, par_front_norm = surrogate_pareto_normalized(2)
par_front_h, par_front_norm = surrogate_pareto_normalized(5)

slopes_l = []
for i in range(par_front_l.shape[0]-1):
    y1, x1 = par_front_l[i,:]
    y2, x2 = par_front_l[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_l.append(m)
slopes_m = []
for i in range(par_front_m.shape[0]-1):
    y1, x1 = par_front_m[i,:]
    y2, x2 = par_front_m[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_m.append(m)
slopes_h = []
for i in range(par_front_h.shape[0]-1):
    y1, x1 = par_front_h[i,:]
    y2, x2 = par_front_h[i+1,:]
    m = (y2-y1)/(x2-x1)
    slopes_h.append(m)

plt.figure(figsize=(5,5))
plt.scatter(np.arange(len(slopes_h)), slopes_h, s=20, facecolors='none', edgecolors='C0', label='High demand')
plt.scatter(np.arange(len(slopes_m)), slopes_m, s=20, facecolors='none', edgecolors='C1', label='Medium demand')
plt.scatter(np.arange(len(slopes_l)), slopes_l, s=20, facecolors='none', edgecolors='C2', label='Low demand')
plt.xlabel("Number of data points", fontsize=13)
plt.ylabel("Slope", fontsize=13)
plt.legend()
plt.grid()
plt.savefig('Slope-comp-lowuniform.pdf')

