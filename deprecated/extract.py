import numpy as np

n=7 #p=0.2 theta=71deg
a=np.array([1.0,0.34018278,0.60965553,0.78622325,0.88760352,0.9406251,0.96526958,0.97237779])
tzop=np.array([ 0.,  0.,1.64027054,0.,  1.03598002,0.,1.12662915])
X1Od=np.array([ 2.,  0.,5.20048201,0.,4.38780868,0.,1.1874829,0.])
X1On=np.array([ 0.,1.42922007,0.,3.02429493,0.,1.89441863,0.,0.30150201])
'''
n=3 #p=0.2 theta=21deg
a=np.array([ 1.        ,  0.30703168,  0.52281303,  0.59863841])
tzop=np.array([ 0.        ,  0.        ,  1.91272968])
X1Od=np.array([ 2.        ,  0.        ,  1.03585267,  0.        ])
X1On=np.array([ 0.        ,  0.99200896,  0.        ,  0.27646272])
'''

caps=np.array([])

B_num=np.copy(X1Od)
B_den=np.copy(X1On)

index=2
while index<n:
    #shunt removal
    BdivL_num=B_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval=np.polyval(BdivL_num,1j*tzop[index])
    BdivL_den_eval=np.polyval(B_den,1j*tzop[index])
    c=BdivL_num_eval/BdivL_den_eval
    caps=np.append(caps,np.real(c))
    
    #update polynomila B
    Lc=np.array([np.real(c),0])  #lambda*c(extracted)
    B_num=np.polysub(B_num,np.polymul(Lc,B_den))
    
    #series removal
    temp_n=np.polymul([1,0],B_num)
    temp_n=np.polydiv(temp_n,[1,0,tzop[index]**2])[0]
    temp_ne=np.polyval(temp_n,1j*tzop[index])
    temp_de=np.polyval(B_den,1j*tzop[index])
    c=temp_ne/temp_de
    caps=np.append(caps,np.real(c))
    
    #update polynomial B
    temp1=np.polymul(B_den,[1,0,tzop[index]**2])
    temp2=np.polymul(B_num,[1/np.real(c),0])
    B_den=np.polysub(temp1,temp2)
    B_num=np.polymul(B_num,[1,0,tzop[index]**2])
    
    
    index+=2

c=np.polyval(B_num[:-1],1)/np.polyval(B_den,1)
caps=np.append(caps,np.real(c))
caps=caps*a[n]
print caps