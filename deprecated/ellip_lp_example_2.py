import math
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

#The example of elliptical lowpass prototype synthesis procedure 
#from Saal and Ulbricht(SU) is realized in this script
 
n=11    #number of section
p=0.2   #reflection coefficient of filter
#w_s=2.790428 #normalized stopband frequency
theta_deg=75

#theta=math.asin(1/w_s) #rad
theta=theta_deg*math.pi/180
theta_deg=180*theta/math.pi #theta degree
# m=(n-1)/2 #EQN(37) p294
m=(n-1)//2 #EQN(37) p294

#complete elliptic integral of the 1st kind K modulus k=sin(theta)
#In scipy library: ellipk(u,m) where m = k^2
K=ss.ellipk(math.sin(theta)**2)
a=np.array([1])   #a[0] is filled with 1 so indices of a[] complies with SU
for v in range(1,n):    
    u=K*v/n
    #EQN(38) notates sn(u,theta)
    #In scipy library: sn(u,m) where m = sin^2(theta)
    e_functs=ss.ellipj(u,pow(math.sin(theta),2))
    sn=e_functs[0]
    a=np.append(a,math.sqrt(math.sin(theta))*sn)       

a=np.append(a,math.sqrt(math.sin(theta)))

#EQN(39) for n odd
delta=1
for u in range(1,m+2):
    delta *= a[2*u-1]**2
delta=1.0/a[n]*delta

#c calculation from page 304
c=1/(delta*math.sqrt(1/p**2 - 1))

#EQN(37) expand for n odd
F=np.array([1,0])
P=np.array([1/c])
for u in range(1,m+1):
    F=np.polymul(F,[1,0,pow(a[2*u],2)])
    P=np.polymul(P,[pow(a[2*u],2),0,1])

#Generate F(jw)*F(-jw) and P(jw)*p(-jw) for n odd
FnF = np.polymul(F, -1*F)
PnP = np.polymul(P, P)
  
#test plot of transfer function
w = np.linspace(0.001,5,num=100000)        
lamb = 1j*w*a[n]            
Kn2 = np.polyval(FnF,lamb)
Kd2 = np.polyval(PnP,lamb) 
K2 = np.real(Kn2)/np.real(Kd2)

S21 = (1/(1+K2))
S21_dB = 10*np.log10(S21)

S11 = 1-S21
S11_dB = 10*np.log10(S11)
plt.clf()
plt.plot(w,S21_dB)
plt.plot(w,S11_dB)

plt.grid()
plt.show()    

#Element extraction                                                                            
#Form (E)(nE)=(F)(nF)+(P)(nP). Page 287. EQN(14), EQN(15)
EnE=np.polyadd(FnF,PnP)

#for n=odd
E = np.array([1])
for root in np.roots(EnE):
    if np.real(root) > 0:
        E = np.polymul(E, np.array([1, root]))

#Form X1o for n=odd
Ee=np.copy(E)
Eo=np.copy(E)
Fe=np.copy(F)
Fo=np.copy(F)

for i in range(len(Ee)):
    if i%2==0:
        Ee[i]=0
        Fe[i]=0
    else:
        Eo[i]=0
        Fo[i]=0
        
X1On=np.polysub(Ee,Fe)
X1Od=np.polyadd(Eo,Fo)   

#sort and organize transmission zeros for finite poles for synthesis
#According SU p296, finite pole frequencies closest to the bandedge should
#be placed in the middle of the filter to avoid negative components.
tz=1/(np.sort(a[2:n:2]))
tzo=tz[::2]
tzo=np.append(tzo,np.sort(tz[1::2]))

#pad zeroes into tzo so the indices match SU
tzop=np.array([0])
for z in tzo:
    tzop=np.append(tzop,[0,z])

caps=np.array([])
inds=np.array([])
omega=np.array([])

B_num=np.copy(X1Od)
B_den=np.copy(X1On)

#element extraction SU p306
index=2
while index<n:
    #shunt removal
    BdivL_num=B_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval=np.polyval(BdivL_num,1j*tzop[index])
    BdivL_den_eval=np.polyval(B_den,1j*tzop[index])
    c=BdivL_num_eval/BdivL_den_eval
    
    #update component and frequency list
    caps=np.append(caps,np.real(c)*a[n])
    omega=np.append(omega,[0])
    inds=np.append(inds,[0])
    
    #update polynomila B
    Lc=np.array([np.real(c),0])  #lambda*c(extracted)
    B_num=np.polysub(B_num,np.polymul(Lc,B_den))
    
    #series removal
    temp_n=np.polymul(np.array([1,0]),B_num)
    temp_n=np.polydiv(temp_n,np.array([1,0,tzop[index]**2]))[0]
    temp_ne=np.polyval(temp_n,1j*tzop[index])
    temp_de=np.polyval(B_den,1j*tzop[index])
    c=temp_ne/temp_de
    
    caps=np.append(caps,np.real(c)*a[n])
    w=tzop[index]/a[n]
    omega=np.append(omega,[w])
    inds=np.append(inds,[1/(w)**2/(np.real(c)*a[n])])
    
    #update polynomial B
    temp1=np.polymul(B_den,np.array([1,0,tzop[index]**2]))
    temp2=np.polymul(B_num,np.array([1/np.real(c),0]))
    B_den=np.polysub(temp1,temp2)
    B_num=np.polymul(B_num,np.array([1,0,tzop[index]**2]))
    
    
    index+=2

c=np.polyval(B_num[:-1],1)/np.polyval(B_den,1)
caps=np.append(caps,np.real(c)*a[n])
inds=np.append(inds,[0])

# print caps
# print inds
print(caps)
print(inds)