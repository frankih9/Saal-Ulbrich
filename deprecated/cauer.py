import math
import numpy as np
import scipy.special as ss
#import matplotlib.pyplot as plt

#The example of elliptical lowpass prototype synthesis procedure 
#from Saal and Ulbricht(SU) is realized in this script
 
n=3    #number of section
p=0.2   #reflection coefficient of filter
#w_s=2.790428 #normalized stopband frequency


#theta=math.asin(1/w_s) #rad
theta=21*math.pi/180
theta_deg=180*theta/math.pi #theta degree
m=(n-1)//2 #EQN(37) p294

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


#test F(lambda)*F(-lambda)
#test P(lambda)*P(-lambda)
FnF=np.array([-1,0,0])
PnP=np.array([1/(c**2)])
for u in range(1,m+1):
    FnF=np.polymul(FnF,[1,0,pow(a[2*u],2)])
    FnF=np.polymul(FnF,[1,0,pow(a[2*u],2)])
    PnP=np.polymul(PnP,[pow(a[2*u],2),0,1])
    PnP=np.polymul(PnP,[pow(a[2*u],2),0,1])

#test E(lambda)*E(-lambda)
EnE=np.polyadd(FnF,PnP)
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
'''   
FnoF=np.array([1,0])
PnoP=np.array([1/c])
for u in range(1,m+1):
    FnoF=np.polymul(FnoF,[-1,0,a[2*u]**2])    
    PnoP=np.polymul(PnoP,[-1*a[2*u]**2,0,1])
'''    
    
#Form F+P. Page 306. Read explaination on why roots for F+P is sufficient
FpP=np.polyadd(F,P)

#Find the only root with no imaginary part
FpP_roots=np.roots(FpP)
for i in range(len(FpP_roots)):
    if np.imag(FpP_roots[i])==0:
        lambda_o=float(np.real(FpP_roots[i]))

#compute p and q.  See EQN(41)
v=-1*lambda_o
p=np.array([1])
q=np.array([1])
for u in range(1,m+1):
    pu=-2*v*math.sqrt((1-pow(a[2*u],2)*pow(a[n],2))*(1-pow(a[2*u],2)/pow(a[n],2)))/(1+pow(v,2)*pow(a[2*u],2))
    p=np.append(p,pu)
    
    qu=(pow(v,2)+pow(a[2*u],2))/(1+pow(v,2)*pow(a[2*u],2))
    q=np.append(q,qu)
    
#Form E. Page 295
#for n=odd
E=np.array([1,-1*lambda_o])
for u in range(1,m+1):
    E=np.polymul(E,[1,-1*p[u],q[u]])

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

'''
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
'''
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

#test for removal
B1n=np.copy(X1Od)
B1d=np.copy(X1On)
#evaluate c1 or B1/lambda @ lambda**2=-(a[2]**2) p306 
#B1/lambda is denoted by B1L
B1Ln=B1n[:-1]
B1Lne=np.polyval(B1Ln,1j*tzop[2])
B1Lde=np.polyval(B1d,1j*tzop[2])
c1=B1Lne/B1Lde

c1L=np.array([np.real(c1),0])

B2n=np.polysub(B1n,np.polymul(B1d,c1L))
B2d=B1d
temp_n=np.polymul([1,0],B2n)
B2nn=np.polydiv(temp_n,[1,0,tzop[2]**2])[0]

temp_ne=np.polyval(B2nn,1j*tzop[2])
temp_de=np.polyval(B2d,1j*tzop[2])
c2=temp_ne/temp_de

B3n=np.polymul(B2n,[1,0,tzop[2]**2])
temp1=np.polymul(B2d,[1,0,tzop[2]**2])
temp2=np.polymul(B2n,[1/np.real(c2),0])
B3d=np.polysub(temp1,temp2)

c3=np.polyval(B3n[:-1],2)/np.polyval(B3d,2)
