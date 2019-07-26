import math
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

#The example of elliptical lowpass prototype synthesis procedure 
#from Saal and Ulbricht(SU) is realized in this script
 
n=5    #number of section
p=0.2   #reflection coefficient of filter
#w_s=2.790428 #normalized stopband frequency


#theta=math.asin(1/w_s) #rad
theta=40*math.pi/180
theta_deg=180*theta/math.pi #theta degree
m=(n-1)/2 #EQN(37) p294

#complete elliptic integral of the 1st kind K modulus k=sin(theta)
#In scipy library: ellipk(u,m) where m = k^2
K=ss.ellipk(pow(math.sin(theta),2))
a=np.array([1])   #a[0] is filled with 1 so indices of a[] complies with SU
for v in range(1,n):    
    u=K*v/n
    #EQN(38) notates sn(u,theta)
    #In scipy library: sn(u,m) where m = sin^2(theta)
    e_functs=ss.ellipj(u,pow(math.sin(theta),2))
    sn=e_functs[0]
    a=np.append(a,math.sqrt(math.sin(theta))*sn)
    #a=np.append(a,sn)   

a=np.append(a,math.sqrt(math.sin(theta)))


#EQN(39) for n odd
delta=1
for u in range(1,m+2):
    delta*=pow(a[2*u-1],2)
delta=1.0/a[n]*delta

#c calculation from page 304
c=1/(delta*math.sqrt(1/pow(p,2)-1))

#EQN(37) expand for n odd
F=np.array([1,0])
P=np.array([1/c])
for u in range(1,m+1):
    F=np.polymul(F,[1,0,pow(a[2*u],2)])
    P=np.polymul(P,[pow(a[2*u],2),0,1])

'''
#Generate F(jw)*F(-jw) and P(jw)*p(-jw) for n odd
FnF=np.array([1,0,0])
PnP=np.array([1/c**2])
for u in range(1,m+1):
    FnF=np.polymul(FnF,[-1,0,pow(a[2*u],2)])
    FnF=np.polymul(FnF,[-1,0,pow(a[2*u],2)])
    PnP=np.polymul(PnP,[-1*pow(a[2*u],2),0,1])
    PnP=np.polymul(PnP,[-1*pow(a[2*u],2),0,1])
''' 
  
FnoF=np.array([1,0])
PnoP=np.array([1/c])
for u in range(1,m+1):
    FnoF=np.polymul(FnoF,[-1,0,(a[2*u])**2])    
    PnoP=np.polymul(PnoP,[-1*(a[2*u])**2,0,1])
    
    



#test plot of transfer function
w=np.linspace(0.1,10,num=1000)
#kn=np.polyval(FnF,w)
#kd=np.polyval(PnP,w)
#kk=kn/kd
kn=np.polyval(FnoF,w)
kd=np.polyval(PnoP,w)
kk=(kn**2)/(kd**2)
#y=10*np.log10(1/(1+(np.polyval(FnF,w)/np.polyval(PnP,w))))
#y=10.*np.log10(1/(1+kk))
y=(1/(1+kk))
g=1-y
yy=10*np.log10(abs(y))
gg=10*np.log10(abs(g))
#plt.plot(w,kn)
#plt.plot(w,kk)
plt.plot(w,yy)
plt.plot(w,gg)
plt.show()    


