###          BIFURCATION DIAGRAMS FOR DG RATE MODEL         ###
###                                                         ###
###  Bifurcation parameters implemented:                    ###
###         - PP synaptic strength                          ### 
###         - Feedback + Feedforward Inhibition synapses    ###
###                                                         ###
### TODO: implement Feedback inhibition analysis only       ###
### ------------------------------------------------------- ###

import numpy as np
import matplotlib.pyplot as plt
from DG_rate import DGRate 
from math import cos, pi

# folder for figure saving
from pathlib import Path
Path("bifrcn").mkdir(exist_ok=True)

# create arrays for PP synaptic strength
input_weak = np.linspace(0,0.01,50)
input_moderate = np.linspace(0.01, 0.1, 100)
input_moderatehigh = np.linspace(0.1, 0.55, 400)
input_high = np.linspace(0.55, 2, 500)

strength = np.concatenate((input_weak, input_moderate, input_moderatehigh, input_high), axis = None)

# array for feedback/feedforward synaptic strength
fffb = np.linspace(0,4,2000) 

# function to allow for easy re-setting of lists
def initialize_lists():
    limcycle_min_peak = []
    limcycle_max_peak = []
    limcycle_min_trough = []
    limcycle_max_trough = []
    return limcycle_min_peak, limcycle_max_peak, limcycle_min_trough, limcycle_max_trough

# initialize lists
limcycle_min_peak, limcycle_max_peak, limcycle_min_trough, limcycle_max_trough = initialize_lists()

# create bifrc'n diagram for PP strength variation condition, keeping FF/FB synapse = 3
for w in strength:
    dg = DGRate('theta', 3, w)
    solns = dg.return_rate_only()
    gc = solns[0]
    limcycle_min_peak.append(min(gc[30500:32250]))  # bounds chosen s.t. only evaluating at theta peak
    limcycle_max_peak.append(max(gc[30500:32250]))
    limcycle_min_trough.append(min(gc[47000:47250]))  # bounds chosen s.t. only evaluating at theta trough
    limcycle_max_trough.append(max(gc[47000:47250]))
    print("working on PPStrength", w) 


fig, axis = plt.subplots(2,1, sharex = True, sharey = True)
axis[0].plot(strength, limcycle_max_peak, color='k')
axis[0].plot(strength, limcycle_min_peak, color='k')
axis[0].set_ylabel('Population Activity')

axis[1].plot(strength, limcycle_max_trough, color='k')
axis[1].plot(strength, limcycle_min_trough, color='k')
axis[1].set_xlabel('PP synaptic strength')
axis[1].set_ylabel('Population Activity')

fig.tight_layout()
fig.savefig('bifrcn/ppstrength.png')

# Re-set lists
limcycle_min_peak, limcycle_max_peak, limcycle_min_trough, limcycle_max_trough = initialize_lists()

# create bifrc'n diagram for FF/FB variation condition, keeping PP synapse = 0.07
for f in fffb:
    dg = DGRate('theta', f, 0.07)
    solns = dg.return_rate_only()
    gc = solns[0]
    limcycle_min_peak.append(min(gc[30500:32250]))
    limcycle_max_peak.append(max(gc[30500:32250]))
    limcycle_min_trough.append(min(gc[47000:47250]))
    limcycle_max_trough.append(max(gc[47000:47250]))
    print("working on fffb PP = 0.07 condition", f)

fig1, axis = plt.subplots(2,1, sharex = True, sharey = True)
axis[0].plot(fffb, limcycle_max_peak, color='k')
axis[0].plot(fffb, limcycle_min_peak, color='k')
axis[0].set_ylabel('Population Activity')

axis[1].plot(fffb, limcycle_max_trough, color='k')
axis[1].plot(fffb, limcycle_min_trough, color='k')
axis[1].set_xlabel('FF/FB synapse strength')
axis[1].set_ylabel('Population Activity')

fig1.tight_layout()
fig1.savefig('bifrcn/fffb_strength_pp0.07.png')

# Re-set lists
limcycle_min_peak, limcycle_max_peak, limcycle_min_trough, limcycle_max_trough = initialize_lists()

# create bifrc'n diagram for FF/FB variation condition, keeping PP synapse = 1.51
for f in fffb:
    dg = DGRate('theta', f, 1.5)
    solns = dg.return_rate_only()
    gc = solns[0]
    limcycle_min_peak.append(min(gc[30500:32250]))
    limcycle_max_peak.append(max(gc[30500:32250]))
    limcycle_min_trough.append(min(gc[47000:47250]))
    limcycle_max_trough.append(max(gc[47000:47250]))
    print("working on fffb PP = 1.5 condition", f)

fig2, axis = plt.subplots(2,1, sharex = True, sharey = True)
axis[0].plot(fffb, limcycle_max_peak, color='k')
axis[0].plot(fffb, limcycle_min_peak, color='k')
axis[0].set_ylabel('Population Activity')

axis[1].plot(fffb, limcycle_max_trough, color='k')
axis[1].plot(fffb, limcycle_min_trough, color='k')
axis[1].set_xlabel('FF/FB synapse strength')
axis[1].set_ylabel('Population Activity')

fig2.tight_layout()
fig2.savefig('bifrcn/fffb_strength_pp1.50.png')
