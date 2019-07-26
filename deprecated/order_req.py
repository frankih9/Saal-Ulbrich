import numpy as np
from SU_Funcs import Kc
from SU_Funcs import As_dB

req_rej = 50
req_ws = 1.15
p = 0.2

theta_deg_init = np.arcsin(1/req_ws)*180/np.pi
#Step 1:
#Estimate starting with n = 3 and step by 2 so n is always odd
n_odd = 3    
rej_dB = As_dB(n_odd, p, theta_deg_init)

while rej_dB < req_rej:
    n_odd = n_odd+2
    rej_dB = As_dB(n_odd, p, theta_deg_init)

rej_odd = rej_dB

#Step 2:
#Does a solution exists for even order or n_odd-1?        
n_even = n_odd -1

#theta_deg increment step
step =0.01

theta_deg = theta_deg_init
_, _, _, wc_p = Kc(n_even, p, theta_deg)
rej_dB = As_dB(n_even, p, theta_deg)
ws_actual = 1/wc_p**2

while (rej_dB > req_rej) and (ws_actual > req_ws): 
    theta_deg = theta_deg + step
    _, _, _, wc_p = Kc(n_even, p, theta_deg)
    rej_dB = As_dB(n_even, p, theta_deg)
    ws_actual = 1/wc_p**2

rej_even = rej_dB

#Step 3:
#Determine if the even order meets requirement
#If not, return the previous odd order results
if rej_even < req_rej:
    req_n = n_odd
    req_theta = theta_deg_init
    actual_rej = rej_odd
    ws_actual = req_ws

else:
    req_n = n_even
    req_theta = theta_deg
    actual_rej = rej_even
    

#ws_theta = 1/np.sin(theta_deg*np.pi/180)
####
print('Required Rejection: ', req_rej)
print('Required ws:        ', req_ws)
print('Required p:         ', p)
print(' ')
#print('Actual Rej Odd:   n=',n_odd,',', rej_odd)
#print('Actual Rej Even:  n=',n_even,',', rej_dB)
#print('current theta_deg:  ', theta_deg)
#print('ws from theta:      ', ws_theta)
print('Required sections:  ', req_n)
print('Required theta_deg: ', req_theta)
print('Actual Ws:          ', ws_actual)
print('Actual Rejection:   ', actual_rej)
