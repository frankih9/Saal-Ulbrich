import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


n=4    #number of section

p=0.333333
theta_deg = 46
#w_s=2.790428 #normalized stopband frequency

#theta=math.asin(1/w_s) #rad
theta = theta_deg*np.pi/180

m=n//2 #EQN(37) p294

wx=1.244
w2=1.630707
w4=1.273134

F=np.polymul(np.array([1,0,0]),np.polymul(np.array([1,0,(wx/w4)**2]), np.array([1,0,(wx/w2)**2])))
P=0.0356362*np.polymul(np.array([1,0,(w4)**2]), np.array([1,0,(w2)**2]))
FnF = np.polymul(F, F)
PnP = np.polymul(P, P)

# #test plot of transfer function
# w = np.linspace(0.1,10,num=1000)        
# s = 1j*w
# #lamb = w*bn           
# Kn2 = np.polyval(FnF,s)
# Kd2 = np.polyval(PnP,s) 
# K2 = np.real(Kn2)/np.real(Kd2)
# 
# S21 = (1/(1+K2))
# S21_dB = 10*np.log10(S21)
# 
# S11 = 1-S21
# S11_dB = 10*np.log10(S11)
# 
# plt.clf()
# plt.plot(w,S21_dB)
# plt.plot(w,S11_dB)
# plt.grid()
# plt.show()

#Element extraction                                                                            
#Form (E)(nE)=(F)(nF)+(P)(nP). Page 287. EQN(14), EQN(15)
EnE = np.polyadd(FnF,PnP)

#for n=even
E = np.array([1])
for root in np.roots(EnE):
    if np.real(root) < 0:
        #print(root)
        E = np.polymul(E, np.array([1, -1*root]))
E = np.real(E)
#print(E)

#zout = (1-p)/(1+p)
zout = (1+p)/(1-p)
t = np.sqrt(1/zout)
pc = 2/(t+1/t)

MnM = np.polysub(EnE,(pc**2)*PnP)
M = np.array([1])
for root in np.roots(MnM):
    if np.real(root) < 0:
        #print(root)
        M = np.polymul(M, np.array([1, -1*root]))
M = np.real(M)

X1On=np.polyadd(E,M)
X1Od=np.polysub(E,M)  

#sort and organize transmission zeros for finite poles for synthesis
#According SU p296, finite pole frequencies closest to the bandedge should
#be placed in the middle of the filter to avoid negative components.

tzop=np.array([0,0,w2,0,w4])


caps=np.array([])
inds=np.array([])
omega=np.array([])

B1_num=np.copy(X1On)
B1_den=np.copy(X1Od)

print('B1 numerator: ', B1_num)
print('B1 denominator: ', B1_den)
print(' ')
bn=1
#element extraction SU p306
index=2
while index < n:
    #shunt removal
    BdivL_num = B1_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval = np.polyval(BdivL_num, 1j*tzop[index])
    BdivL_den_eval = np.polyval(B1_den, 1j*tzop[index])
    c = BdivL_num_eval/BdivL_den_eval
    
    #update component and frequency list
    caps = np.append(caps, np.real(c)*bn)
    omega = np.append(omega,[0])
    inds = np.append(inds,[0])
    
    #update polynomila B
    Lc = np.array([np.real(c), 0])  #lambda*c(extracted)
    B2_num = np.polysub(B1_num, np.polymul(Lc, B1_den))
    B2_den = np.copy(B1_den)
    print('B2 numerator: ', B2_num)
    print('B2 denominator: ', B2_den)
    print(' ')
    
    #series removal
    temp_n = np.polymul(np.array([1,0]), B_num)
    
    #temp_n = np.polydiv(temp_n, np.array([1,0,tzop[index]**2]))[0]
    B_den = np.polymul(B_den, np.array([1,0,tzop[index]**2]))
    
        
    temp_ne = np.polyval(temp_n, 1j*tzop[index])
    print(temp_ne)
    temp_de = np.polyval(B_den, 1j*tzop[index])
    print(temp_de)
    c = temp_ne/temp_de
    
    caps=np.append(caps,np.real(c)*bn)
    w=tzop[index]/bn
    omega=np.append(omega,[w])
    inds=np.append(inds,[1/(w)**2/(np.real(c)*bn)])
    
    #update polynomial B
    temp1=np.polymul(B_den,np.array([1,0,tzop[index]**2]))
    temp2=np.polymul(B_num,np.array([1/np.real(c),0]))
    B_den=np.polysub(temp1,temp2)
    B_num=np.polymul(B_num,np.array([1,0,tzop[index]**2]))
    
    print('B3 numerator: ', B_num)
    print('B3 denominator: ', B_den)
    print(' ')
       
    index+=2

#shunt removal
BdivL_num = B_num[:-1] #B divided by lambda when constant term of B is 0
BdivL_num_eval = BdivL_num[0]
BdivL_den_eval = B_den[0]
c = BdivL_num_eval/BdivL_den_eval

caps=np.append(caps, np.real(c)*bn)
inds=np.append(inds, [0])

Lc = np.array([np.real(c), 0])  #lambda*c(extracted)
B_num = np.polysub(B_num, np.polymul(Lc, B_den))

print('B4 numerator: ', B_num)
print('B4 denominator: ', B_den)
print(' ')

# #see table 4.4 page 129 in Zverev for X2O
# l = (Ee[0]+Fe[0])/(Eo[0]+Fo[0])
# caps=np.append(caps, 0)
# inds=np.append(inds, l*bn*zout)

# print caps
# print inds
print(caps)
print(inds)