import numpy as np
import pandas as pd
from SU_Funcs import Ka
from SU_Funcs import Kb
from SU_Funcs import Kc
from SU_Funcs import E
from SU_Funcs import Extract_Order
from SU_Funcs import Extract
from SU_Funcs import As_dB
from SU_Funcs import Plot_S
from SU_Funcs import Eval_K
from SU_Funcs import Eval_Elements
from SU_Funcs import Element_Table
from SU_Funcs import Find_NTheta

##
#Define filter requirements. Change me!
##
req_rej = 48
req_ws = 1.5
req_p = 0.2 

##
#Find the required order and theta
##
n, theta_deg, cauer_type = Find_NTheta(req_rej, req_ws, req_p)
p = req_p

##
#Formulate the polynomial K(lambda)
##
K = {'a': Ka(n, p, theta_deg), 'b': Kb(n, p, theta_deg), 'c': Kc(n, p, theta_deg)}
F, P, co_tz, wc_p = K[cauer_type]

##
#Formulate the Hurwitz polynomial E(lambda)
##
E, FnF, PnP = E(F, P)

##
#Frequency range for analysis
##
wstep = .02
w = np.arange(wstep, 10, wstep)

##
#Calculate stop band frequency and rejection level
##
ws = 1/wc_p**2
rej_dB = As_dB(n, p, theta_deg)

##
#Evaluate K(lambda) and plot associated S11 and S21
##
S11_dB, S21_dB = Eval_K(FnF, PnP, w, wc_p)
title = 'Filter response from polynomial $K({\lambda})$'
Plot_S(S11_dB, S21_dB, w, ws, rej_dB, title, 1)

##
#Extract circuit elements(caps, inds, and terms)
##
tzop = Extract_Order(co_tz)
cap_arr, ind_arr, omega_arr, term_arr = Extract(n, p, tzop, E, F, wc_p, cauer_type)

##
#Simulate circuit with extracted elements
##
S11_dB, S21_dB = Eval_Elements(cap_arr, ind_arr, term_arr, w)
title = 'Filter response from extracted capacitors and inductors'
Plot_S(S11_dB, S21_dB, w, ws, rej_dB, title, 2)

##
#Print a dataframe of all components
##
cap_tab, ind_tab, omega_tab, term_tab = Element_Table(cap_arr, ind_arr, omega_arr, term_arr)
data = [cap_tab, ind_tab, omega_tab, term_tab] 
#ind = ['$C$','$L$','${\Omega}$','${R}$']
labs = ['Caps','Inds','Omegas','Terms']
df = pd.DataFrame(data, columns = np.arange(0, n+2), index = labs)
#df.replace(0, '', inplace = True)
df.replace(0, '--', inplace = True)
df = df.T
df.style
df.round(1)
print(df)
