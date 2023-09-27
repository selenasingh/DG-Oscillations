import numpy as np
from math import cos, pi

# nicer font options:
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # computer modern 
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12}) #12 rasters

import matplotlib.pyplot as plt

from pathlib import Path
Path("figures").mkdir(exist_ok=True)

# Numerical Integration of a 4D wilson-cowan system, simulating dentate gyrus population rate activity

class DGRate(object):
    def __init__(self,
                 PP_freq,       # frequency of PP inputs
                 fbi,           # scale feedback and feedforward inhibition synaptic weights (between GCs and BCs)
                 PP_weight,     # scale PP synaptic weight (onto GCs and BCs)
                ):
        self.pars = {}
        self.pars['input_freq'] = PP_freq
        self.pars['wPPg'] = PP_weight
        self.pars['wgb'] = fbi
        self.pars['wbg'] = fbi

        self.flag = PP_freq + '_' + str(fbi) + '_' + str(PP_weight)

        self.plot_dg_rates()
        self.plot_cell_fi()
        self.plot_GC_rates()

    def parameters(self, **kwargs):

        # gc parameters
        self.pars['tau_g'] = 3.1  # membrane timescale of granule cell [ms]
        self.pars['gain_g'] = 60  # gain of granule cell 
        self.pars['thresh_g'] = 0.055  # threshold of granule cell

        # bc parameters
        self.pars['tau_b'] = 1.0  # membrane timescale of basket cell [ms]
        self.pars['gain_b'] = 250  # gain of basket cell
        self.pars['thresh_b'] = 0.025  # threshold of basket cell

        # mc parameters
        self.pars['tau_m'] = 3.5  # membrane timescale of mossy cell [ms]
        self.pars['gain_m'] = 25  # gain of mossy cell
        self.pars['thresh_m'] = 0.005  # threshold of mossy cell

        # hc parameters
        self.pars['tau_h'] = 1.5  # membrane timescale of hipp cell [ms]
        self.pars['gain_h'] = 20  # gain of hipp cell
        self.pars['thresh_h'] = 0  # threshold of hipp cell

        # synaptic weights
        self.pars['wgg'] = 0.  # GC to GC   ; mossy fiber "sprouting"
        self.pars['wmg'] = 1.  # MC to GC
        self.pars['wbg'] = 3  # BC to GC ; 1 for lesion study
        self.pars['whg'] = 1.  # HC to GC
        self.pars['wbb'] = 1.  # BC to BC
        self.pars['wgb'] = 3.  # GC to BC ; 0 for lesion study
        self.pars['wmb'] = 1.  # MC to BC
        self.pars['whb'] = 1.  # HC to BC
        self.pars['wmm'] = 1.  # MC to MC
        self.pars['wgm'] = 1.  # GC to MC
        self.pars['wbm'] = 1.  # BC to MC
        self.pars['whm'] = 1.  # HC to MC
        self.pars['wmh'] = 1.  # MC to HC
        self.pars['wgh'] = 1.  # GC to HC

        self.pars['wPPg'] = 1  # scale PP synaptic weight to gcs 
        self.pars['wPPb'] = self.pars['wPPg']/2  # scale PP synaptic input to bcs

        # integration parameters
        self.pars['T'] = 1000.  # Total duration of simulation [ms]
        self.pars['dt'] = .01  # Simulation time step [ms]
        self.pars['g_init'] = 0.001  # Initial value of granule cells
        self.pars['b_init'] = 0.001  # Initial value of basket cells
        self.pars['m_init'] = 0.001  # Initial value of mossy cells
        self.pars['h_init'] = 0.001  # Initial value of hipp cells

        # External parameters if any
        for k in kwargs:
            self.pars[k] = kwargs[k]

        # Vector of  time points [ms]
        self.pars['range_t'] = np.arange(0, self.pars['T'], self.pars['dt'])

        # PP input
        if self.pars['input_freq'] == 'theta':
            cos_scale = 0.02  # 3 Hz
        elif self.pars['input_freq'] == 'alpha':
            cos_scale = 0.08  # 12 Hz
        elif self.pars['input_freq'] == 'gamma':
            cos_scale = 0.2  # 35 Hz
        elif self.pars['input_freq'] == 'delta':
            cos_scale = 0.005
        elif self.pars['input_freq'] == 'constant':
            cos_scale = 0 # 0 Hz, constant of 1 

        periodic_forcing = []
        for x in self.pars['range_t']:
            periodic_forcing.append((1 + cos(cos_scale * x))/2) #remove 2, get bifc'n 

        self.pars['PP'] = periodic_forcing

        return self.pars

    def F(self, i, gain, thresh):
        """
        Population activation function, F-I curve

        Args:
          i     : the population input
          gain  : the gain of the function
          thresh : the threshold of the function

        Returns:
          f     : the population activation response f(x) for input x
        """

        # add the expression of f = F(x)
        f = (1 + np.exp(-gain * (i - thresh))) ** -1 - (1 + np.exp(gain * thresh)) ** -1

        return f

    def simulate_DG(self,
                    # gc
                    tau_g, gain_g, thresh_g,
                    # bc
                    tau_b, gain_b, thresh_b,
                    # mc
                    tau_m, gain_m, thresh_m,
                    # hc
                    tau_h, gain_h, thresh_h,

                    # synaptic weights
                    wgg, wmg, wbg, whg, wbb, wgb, wmb, whb, wmm, wgm, wbm, whm, wmh, wgh,

                    # perforant path
                    wPPg, wPPb, PP,

                    # simulation params
                    range_t, dt, g_init, b_init, m_init, h_init,

                    **other_pars):
        """
            Simulate two sets of Wilson-Cowan equations, modelling dentate gyrus population activity

            Args:
              Parameters of the 4D system

            Returns:
              g, b, m, h (arrays) : Activity of granule, basket, mossy and hipp cells.
            """
        # Initialize activity arrays
        Lt = range_t.size
        g = np.append(g_init, np.zeros(Lt - 1))
        b = np.append(b_init, np.zeros(Lt - 1))
        m = np.append(m_init, np.zeros(Lt - 1))
        h = np.append(h_init, np.zeros(Lt - 1))

        # Simulate the 4D system
        for k in range(Lt - 1):
            # Calculate the derivative of the granule cell population
            dg = dt / tau_g * (-g[k] + self.F(wgg * g[k] + wmg * m[k] - wbg * b[k] - whg * h[k] + wPPg * PP[k],
                                              gain_g, thresh_g))

            # Calculate the derivative of basket cell population
            db = dt / tau_b * (-b[k] + self.F(-wbb * b[k] + wgb * g[k] + wmb * m[k] - whb * h[k] + wPPb * PP[k],
                                              gain_b, thresh_b))

            # Calculate the derivative of the mossy cell population
            dm = dt / tau_m * (-m[k] + self.F(wmm * m[k] + wgm * g[k] - wbm * b[k] - whm * h[k],
                                              gain_m, thresh_m))

            # Calculate the derivative of the hipp cell population
            dh = dt / tau_h * (-h[k] + self.F(wmh * m[k] + wgh * g[k],
                                              gain_h, thresh_h))

            # Update using Euler's method
            g[k + 1] = g[k] + dg
            b[k + 1] = b[k] + db
            m[k + 1] = m[k] + dm
            h[k + 1] = h[k] + dh

        return g, b, m, h

    def plot_dg_rates(self):
        params = self.parameters()
        g, b, m, h = self.simulate_DG(**params)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(4,5))

        ax1.plot(params['range_t'], h, color='0.5', label='HIPP')
        ax1.set_ylabel("HIPP")

        ax2.plot(params['range_t'], b, color='0.5', label='BC')
        ax2.set_ylabel("BC")

        ax3.plot(params['range_t'], m, color='0.5', label='MC')
        ax3.set_ylabel("MC")

        ax4.plot(params['range_t'], g, color='0.5', label='GC')
        ax4.set_ylabel("GC")

        ax5.plot(params['range_t'], params['PP'], color='0.5', label='PP')
        ax5.set_ylabel("PP")
        ax5.set_xlabel("Time (ms)")

        fig.tight_layout()
        fig.align_ylabels()
        fig.savefig('figures/population_rates_%s.pdf' % self.flag, bbox_inches="tight")
        plt.close()

    def plot_GC_rates(self):
        params = self.parameters(T = 750)
        g, b, m, h = self.simulate_DG(**params)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,4), gridspec_kw={'height_ratios': [4, 1]})

        ax1.plot(params['range_t'], g, color='0.5', label='GC')
        ax1.set_ylabel("DG")

        ax2.plot(params['range_t'], params['PP'], color='0.5', label='PP')
        ax2.set_ylabel("PP")
        ax2.set_xlabel("Time (ms)")

        fig.tight_layout()
        fig.align_ylabels()
        plt.setp(ax1.get_xticklabels(), visible=False)
        fig.savefig('figures/GC_rates_%s.pdf' % self.flag, bbox_inches="tight")
        plt.close()

    def plot_cell_fi(self):
        params = self.parameters()
        currs = np.linspace(0, 0.06, 20)

        g_fi = []
        g_fi_m = []
        g_fi_h= []
        m_fi = []
        h_fi = []
        b_fi = []

        for current in currs:
            g_f = self.F(current, params['gain_g'], params['thresh_g'])
            g_f_m = self.F(current, (60*1.05), (0.055*0.95))
            g_f_h = self.F(current, (60*0.90), (0.055*1.1))
            m_f = self.F(current, params['gain_m'], params['thresh_m'])
            h_f = self.F(current, params['gain_h'], params['thresh_h'])
            b_f = self.F(current, params['gain_b'], params['thresh_b'])

            g_fi.append(g_f)
            g_fi_m.append(g_f_m)
            g_fi_h.append(g_f_h)
            m_fi.append(m_f)
            h_fi.append(h_f)
            b_fi.append(b_f)

        plt.figure(figsize=(6,6))
        plt.plot(currs, g_fi, color = 'k', linestyle = 'solid', label='GC')
        plt.plot(currs, g_fi_m, color = 'red', linestyle = 'solid', label='GC +')
        plt.plot(currs, g_fi_h, color = 'blue', linestyle = 'solid', label='GC -')
        plt.plot(currs, h_fi, color = 'k', linestyle = 'dashdot', label='HIPP')
        plt.plot(currs, m_fi, color = 'k', linestyle = 'dotted', label='MC')
        plt.plot(currs, b_fi, color = 'k', linestyle = 'dashed', label='BC')
        plt.legend()
        plt.xlabel('Population Input')
        plt.ylabel('Population Response')
        plt.savefig('figures/all_fi.pdf')
        plt.close()
    
    def return_rate_only(self):
        params = self.parameters()
        g, b, m, h = self.simulate_DG(**params)
        return g, b, m, h

oscillations = ['theta', 'alpha', 'gamma', 'delta']
strength = np.linspace(0,2,30)
gain = np.linspace(0, 10, 30)
fbi = np.linspace(0,4,40)

# single runs 
'''
for freq in oscillations:
     dg_1 = DGRate(freq, 3, 0.07)
     dg_2 = DGRate(freq, 0.25, 1.51)
     dg_3 = DGRate(freq, 3, 1.75)
     dg_4 = DGRate(freq, 0.25, 0.07)
     dg_5 = DGRate(freq, 3.75, 0.07)
'''

#dg_1 = DGRate('theta', 0.25, 1.51)
# studying bifurcations:
'''
for w in strength:      #PP synaptic strength
    for f in fbi:       #FB/FF synaptic strength
        dg = DGRate('theta', f, w) 
        print("testing params of", f, w)
'''
'''
# gamma unstable regime 
for freq in oscillations:
    dg = DGRate(freq, 1.65, 1.65)

# gamma stable regime
for freq in oscillations:
    dg = DGRate(freq, 2.05, 0.207)
    #dg = DGRate(freq, 3, 0.2)
'''
