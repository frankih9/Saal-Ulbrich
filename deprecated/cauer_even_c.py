import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
import pandas as pd

#The example of elliptical lowpass prototype synthesis procedure 
#from Saal and Ulbricht(SU) is realized in this script
 
n=6    #number of section
p=0.2   #reflection coefficient of filter
typ = 'c'

zout = typ!='b'

theta_deg = 46

#odd=n%2

#theta=math.asin(1/w_s) #rad
theta = theta_deg*np.pi/180
#Wso=1/np.sin(theta)

m=n//2 #EQN(37) p294

#complete elliptic integral of the 1st kind K modulus k=sin(theta)
#In scipy library: ellipk(u,m) where m = k^2
K=ss.ellipk(pow(np.sin(theta),2))
a=np.array([1])   #a[0] is filled with 1 so indices of a[] complies with SU
for v in range(1,n):    
    u=K*v/n
    #EQN(38) notates sn(u,theta)
    #In scipy library: sn(u,m) where m = sin^2(theta)
    e_functs=ss.ellipj(u, pow(np.sin(theta),2))
    sn=e_functs[0]
    a=np.append(a, np.sqrt(np.sin(theta))*sn)
       

a=np.append(a, np.sqrt(np.sin(theta)))

cc=np.array([1])
for v in range(1,n):
    cc = np.append(cc, np.sqrt((a[v]**2-a[1]**2)/(1-(a[v]**2)*(a[1]**2))))   

ws=1/a[n-1]**2
       
#EQN(39) for n even case a 
delta=1
for u in range(1,m+1):
    delta *= a[2*u-1]**2

#c calculation from page 304
c=1/(delta*np.sqrt(1/p**2-1))

#EQN(37) expand for n even
F = np.array([1, 0, 0])
P = np.array([1/c])
for u in range(2,m+1):
    F = np.polymul(F, np.array([1, 0, cc[2*u-1]**2]))
    P = np.polymul(P, np.array([cc[2*u-1]**2, 0, 1]))


#Generate F(jw)*F(-jw) and P(jw)*p(-jw) for n even
FnF = np.polymul(F, F)
PnP = np.polymul(P, P)

#test plot of transfer function
w = np.linspace(0.0001,10,num=100000)        
lamb = 1j*w*a[n-1] 
   
Kn2 = np.polyval(FnF,lamb)
Kd2 = np.polyval(PnP,lamb) 
K2 = np.real(Kn2)/np.real(Kd2)

S21 = (1/(1+K2))
S21_dB = 10*np.log10(S21)

As = S21_dB[int(ws*10000)-1]

S11 = 1-S21
S11_dB = 10*np.log10(S11)

plt.clf()
plt.plot(w,S21_dB)
plt.plot(w,S11_dB)
plt.plot(w, As*np.ones(len(w)),linestyle=':')
plt.grid()
plt.show()

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

#Form X1o for n=even
Ee=np.copy(E)
Eo=np.copy(E)
Fe=np.copy(F)
Fo=np.copy(F)
for i in range(len(Ee)):
    if i%2==0:
        Eo[i]=0
        Fo[i]=0
    else:
        Ee[i]=0
        Fe[i]=0
Eo=Eo[1:]
Fo=Fo[1:]

X1On=np.polysub(Ee,Fe)
X1On = X1On[2:]
X1Od=np.polyadd(Eo,Fo)  

#sort and organize transmission zeros for finite poles for synthesis
#According SU p296, finite pole frequencies closest to the bandedge should
#be placed in the middle of the filter to avoid negative components.
bz = np.array([])
for u in range(2,m+1):
    bz = np.append(bz, np.array([cc[2*u-1]]))

tz=1/(np.sort(bz))
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
while index < n:
    #shunt removal
    BdivL_num = B_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval = np.polyval(BdivL_num, 1j*tzop[index])
    BdivL_den_eval = np.polyval(B_den, 1j*tzop[index])
    c = BdivL_num_eval/BdivL_den_eval
    
    #update component and frequency list
    caps = np.append(caps, np.real(c)*a[n-1])
    omega = np.append(omega,[0])
    inds = np.append(inds,[0])
    
    #update polynomila B
    Lc = np.array([np.real(c), 0])  #lambda*c(extracted)
    B_num = np.polysub(B_num, np.polymul(Lc, B_den))
        
    #series removal
    temp_n = np.polymul(np.array([1,0]), B_num)
    temp_n = np.polydiv(temp_n, np.array([1,0,tzop[index]**2]))[0]
            
    temp_ne = np.polyval(temp_n, 1j*tzop[index])
    temp_de = np.polyval(B_den, 1j*tzop[index])
    c = temp_ne/temp_de
    
    caps=np.append(caps,np.real(c)*a[n-1])
    w=tzop[index]/a[n-1]
    omega=np.append(omega,[w])
    inds=np.append(inds,[1/(w)**2/(np.real(c)*a[n-1])])
    
    #update polynomial B
    temp1=np.polymul(B_den,np.array([1,0,tzop[index]**2]))
    temp2=np.polymul(B_num,np.array([1/np.real(c),0]))
    B_den=np.polysub(temp1,temp2)
    B_num=np.polymul(B_num,np.array([1,0,tzop[index]**2]))
    
    index+=2

#shunt removal
BdivL_num = B_num[:-1] #B divided by lambda when constant term of B is 0
BdivL_num_eval = BdivL_num[0]
BdivL_den_eval = B_den[0]
c = BdivL_num_eval/BdivL_den_eval

caps=np.append(caps, np.real(c)*a[n-1])
inds=np.append(inds, [0])

Lc = np.array([np.real(c), 0])  #lambda*c(extracted)
B_num = np.polysub(B_num, np.polymul(Lc, B_den))

#see table 4.4 page 129 in Zverev for X2O
l = (Ee[0]+Fe[0])/(Eo[0]+Fo[0])
caps=np.append(caps, 0)
inds=np.append(inds, l*a[n-1])

caps=np.append(np.array([0]), caps)
caps=np.append(caps, 0)
inds=np.append(np.array([0]), inds)
inds=np.append(inds, 0)
term=np.array([1])
term=np.append(term, np.zeros(n))
term=np.append(term, zout)
omega=np.append(omega, np.zeros(n-len(omega)+1))
omega=np.append(np.array([0]), omega)

print(caps)
print(inds)

dict = {'2. Caps':caps,
        '3. Inds': inds,
        '1. Res': term,
        '4. O': omega}
# creating a dataframe from a dictionary 
df = pd.DataFrame(dict)
df.style
print(df)