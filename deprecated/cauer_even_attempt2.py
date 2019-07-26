import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

#The example of elliptical lowpass prototype synthesis procedure 
#from Saal and Ulbricht(SU) is realized in this script
 
n=6    #number of section
p=0.2   #reflection coefficient of filter
zout = (1-p)/(1+p)

theta_deg = 41
#w_s=2.790428 #normalized stopband frequency

#odd=n%2

#theta=math.asin(1/w_s) #rad
theta = theta_deg*np.pi/180

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

#EQN(44) for n even case b page 295-296
w = np.sqrt((1-(a[1]**2)*(a[n]**2))*(1-(a[1]/a[n])**2))

bP=np.array([1])
bS=np.array([1])
for v in range(1,n):
    bP = np.append(bP, np.sqrt(w/(a[v]**-2 - a[1]**2)))
    bS = np.append(bS, np.sqrt((a[v]**2 - a[1]**2)/w))

bn = np.sqrt(a[n]*a[n-1])
        
#EQN(39) for n even
delta_b = 1
for u in range(1,m+1):
    delta_b *= bP[2*u-1]**2


#c calculation from page 304
c = 1/(delta_b*np.sqrt(1/(p**2) - 1))

#EQN(37) expand for n even
F = np.array([1, 0, (bP[1]**2)])
P = np.array([1/c])
for u in range(2,m+1):
    F = np.polymul(F, np.array([1, 0, bP[2*u-1]**2]))
    P = np.polymul(P, np.array([bS[2*u-1]**2, 0, 1]))


#Generate F(jw)*F(-jw) and P(jw)*p(-jw) for n even
#nF=np.copy(F)
#nF[0:-1:4]=nF[0:-1:4]*-1
FnF = np.polymul(F, F)

#nP=np.copy(P)
#nP[2]=nP[2]*-1
PnP = np.polymul(P, P)

#test plot of transfer function
w = np.linspace(0.001,10,num=10000)        
lamb = 1j*w*bn 
#lamb = w*bn           
Kn2 = np.polyval(FnF,lamb)
Kd2 = np.polyval(PnP,lamb) 
K2 = np.real(Kn2)/np.real(Kd2)

S21 = (1/(1+K2))
S21_dB = 10*np.log10(S21)

S11 = 1-S21
S11_dB = 10*np.log10(S11)
AsW = 1/np.sqrt(np.sin(theta))/bn
plt.clf()
plt.plot(AsW, S21_dB[int(1000*AsW)-1], '-gD')
plt.plot(w,S21_dB)
plt.plot(w,S11_dB)
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

#X1On=np.polysub(Ee,Fe)
#X1Od=np.polysub(Eo,Fo)

#sort and organize transmission zeros for finite poles for synthesis
#According SU p296, finite pole frequencies closest to the bandedge should
#be placed in the middle of the filter to avoid negative components.
bz = np.array([])
for u in range(2,m+1):
    bz = np.append(bz, np.array([bS[2*u-1]]))
#bz = np.append(bz,np.inf)

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

print('B1 numerator: ', B_num)
print('B1 denominator: ', B_den)
print(' ')

#element extraction SU p306
index=2
while index < n:
    #shunt removal
    BdivL_num = B_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval = np.polyval(BdivL_num, 1j*tzop[index])
    BdivL_den_eval = np.polyval(B_den, 1j*tzop[index])
    c = BdivL_num_eval/BdivL_den_eval
    
    #update component and frequency list
    caps = np.append(caps, np.real(c)*bn)
    omega = np.append(omega,[0])
    inds = np.append(inds,[0])
    
    #update polynomila B
    Lc = np.array([np.real(c), 0])  #lambda*c(extracted)
    B_num = np.polysub(B_num, np.polymul(Lc, B_den))
    #B_den = np.copy(B_den)
    print('B2 numerator: ', B_num)
    print('B2 denominator: ', B_den)
    print(' ')
    
    #series removal
    temp_n = np.polymul(np.array([1,0]), B_num)
    print(np.roots(temp_n))
    print(np.roots(np.array([1,0,tzop[index]**2])))
    print(np.polydiv(temp_n, np.array([1,0,tzop[index]**2])))
    print(' ')
    temp_n = np.polydiv(temp_n, np.array([1,0,tzop[index]**2]))[0]
    print(temp_n)
        
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

#see table 4.4 page 129 in Zverev for X2O
l = (Ee[0]+Fe[0])/(Eo[0]+Fo[0])
caps=np.append(caps, 0)
inds=np.append(inds, l*bn*zout)

# print caps
# print inds
print(caps)
print(inds)