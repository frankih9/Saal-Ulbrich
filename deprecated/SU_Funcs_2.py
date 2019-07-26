import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


def a_v(n, theta_deg):
    '''
    Calculate a[v] and delta values for both
    odd and even orders of type 'a' cauer 
    lowpass filters.  The equations are from
    equations (37), (38), and (39) of Saal and Ulbrich 
    paper on page 294.
    
    n = number of sections or lowpass order
    p = reflection coefficient or % of reflection
    theta_deg: theta in degrees such that sin(theta)=wc/ws. See page 292
    '''

    theta = theta_deg*np.pi/180 #radians

    #m calculation, see EQN(37) p294
    if n%2 == 0:
        #n is even
        m = n//2 
    else:
        #n is odd
        m = (n-1)//2 #

    #Complete elliptic integral of the 1st kind K with modulus k=sin(theta)
    #In scipy library: ellipk(u,m) where m = k^2
    K = ss.ellipk(np.sin(theta)**2)

    a = np.array([1])   #a[0] is filled with 1 so indices of a[] complies with SU
    for v in range(1, n):  #1 to n-1 inclusive  
        u = K*v/n
        #EQN(38) notates sn(u,theta)
        #In scipy library: sn(u,m) where m = sin^2(theta)
        jac_ell = ss.ellipj(u, np.sin(theta)**2)
        sn = jac_ell[0]
        a = np.append(a, np.sqrt(np.sin(theta))*sn)
        
    a = np.append(a, np.sqrt(np.sin(theta))) #append a[n]
    
    wc_pa = a[n]
    
    #delta calculation, see EQN(39)
    delta_a = 1

    if n%2 == 0:
        m_end = m
        scale = 1.0
    else:
        m_end = m+1
        scale = 1.0/a[n]
    
    for u in range(1,m_end+1):
        delta_a *= a[2*u-1]**2
    
    delta_a = scale*delta_a
    
    return a, delta_a, wc_pa

    
def b_v(n, theta_deg):
    '''
    Calculate b[v] and delta values for even order 
    type 'b' cauer lowpass filters.  The equations 
    are from equation (44) of Saal and Ulbrich paper 
    on pages 295-296.
    
    n = number of sections or lowpass order
    p = reflection coefficient or % of reflection
    theta_deg: theta in degrees such that sin(theta)=wc/ws. See page 292
    For even orders, theta_deg is not equal to wc/ws.
    '''
    a, delta_a, _ = a_v(n, theta_deg)
    
    #EQN(44) for n even case b page 295-296
    w = np.sqrt((1-(a[1]**2)*(a[n]**2))*(1-(a[1]/a[n])**2))
    
    bP = np.array([1])
    bS = np.array([1])
    for v in range(1,n):
        bP = np.append(bP, np.sqrt(w/(a[v]**-2 - a[1]**2)))
        bS = np.append(bS, np.sqrt((a[v]**2 - a[1]**2)/w))

    bn = np.sqrt(a[n]*a[n-1])

    wc_pb = bn
    
    #m calculation, see EQN(37) p294
    #n is even
    m = n//2 
       
    #EQN(44) for n even
    delta_b = 1
    for u in range(1,m+1):
        delta_b *= bP[2*u-1]**2       
                        
    return bP, bS, delta_b, wc_pb

                                                        
def c_v(n, theta_deg):
    '''
    Calculate c[v] and delta values for even order 
    type 'c' cauer lowpass filters.  The equations 
    are from equation (47) of Saal and Ulbrich paper 
    on page 296.
    
    n = number of sections or lowpass order 
    p = reflection coefficient or % of reflection
    theta_deg: theta in degrees such that sin(theta)=wc/ws. See page 292
    For even orders, theta_deg is not equal to wc/ws.
    '''
    #delta_c equals delta_a. See right top of page 296
    a, delta_c, _ = a_v(n, theta_deg)
    
    c = np.array([1])
    for v in range(1,n):
        # EQN(47) on page 296
        c = np.append(c, np.sqrt((a[v]**2-a[1]**2)/(1-(a[v]**2)*(a[1]**2))))
        
    wc_pc = a[n-1]
    
    return c, delta_c, wc_pc
    

def Ka(n, p, theta_deg):
    #n is odd, EQN(37) p294
    m = (n-1)//2
    
    a, delta_a, wc_p = a_v(n, theta_deg)
    
    #const_c calculation from page 304
    const_c = 1/(delta_a*np.sqrt(1/p**2 - 1)) 
    
    #EQN(37) expand for n odd
    F = np.array([1,0])
    P = np.array([1/const_c])
    
    #These are the inverse of the transmission zero frequencies
    co_tz= np.array([])
    
    for u in range(1,m+1):
        #form numerator of K(lambda) or F(lambda)
        F = np.polymul(F, [1, 0, a[2*u]**2])
        
        #form denominator of K(lambda) or P(lambda)
        P = np.polymul(P, [a[2*u]**2, 0, 1])
        
        #collect the coefficient a[v] used in P(lambda)
        co_tz = np.append(co_tz, a[2*u])
               
    return F, P, co_tz, wc_p


def Kb(n, p, theta_deg):
    #EQN(39) for n even
    m = n//2
     
    bP, bS, delta_b, wc_p = b_v(n, theta_deg)
    
    #c calculation from page 304
    const_c = 1/(delta_b*np.sqrt(1/p**2-1))

    #EQN(43) expand for n even
    F = np.array([1, 0, (bP[1]**2)])
    P = np.array([1/const_c])
    
    #These are the inverse of the transmission zero frequencies
    co_tz = np.array([])
    
    for u in range(2,m+1):
        #form numerator of K(lambda) or F(lambda)
        F = np.polymul(F, np.array([1, 0, bP[2*u-1]**2]))
        
        #form denominator of K(lambda) or P(lambda)
        P = np.polymul(P, np.array([bS[2*u-1]**2, 0, 1]))
        
        #collect the constants bS[2*u-1] used in P(lambda)
        co_tz = np.append(co_tz, bS[2*u-1])
    
    return F, P, co_tz, wc_p


def Kc(n, p, theta_deg):
    #n is even EQN(37) p294
    m = n//2
     
    c, delta_c, wc_p = c_v(n, theta_deg)
    
    #const_c calculation from page 304
    const_c = 1/(delta_c*np.sqrt(1/p**2-1))
    
    #EQN(46) expand for n even
    F = np.array([1, 0, 0])
    P = np.array([1/const_c])
    
    #These are the inverse of the transmission zero frequencies
    co_tz = np.array([])
    
    for u in range(2,m+1):
        #form numerator of K(lambda) or F(lambda)
        F = np.polymul(F, np.array([1, 0, c[2*u-1]**2]))
        
        #form denominator of K(lambda) or P(lambda)
        P = np.polymul(P, np.array([c[2*u-1]**2, 0, 1]))
        
        #collect the coefficient a[v] used in P(lambda)
        co_tz = np.append(co_tz, c[2*u-1])

    return F, P, co_tz, wc_p


def E(F, P):
    '''
    Forms the Hurwitz polynomial E(lambda)
    '''
    #Generate F(jw)*F(-jw) and P(jw)*p(-jw) for n even
    #nX = X(-jw).  Changes sign of coefficients of odd order only.
    nF = np.copy(F)
    nP = np.copy(P)
    #Odd order indexing from penultimate element while stepping by 2 in reverse. 
    nF[-2::-2] = -1*nF[-2::-2]
    nP[-2::-2] = -1*nP[-2::-2]
    
    FnF = np.polymul(F, nF)
    PnP = np.polymul(P, nP)
            
    #Form (E)(nE)=(F)(nF)+(P)(nP). Page 287. EQN(14), EQN(15)
    EnE = np.polyadd(FnF,PnP)

    E = np.array([1])
    for root in np.roots(EnE):
        if np.real(root) < 0:
            E = np.polymul(E, np.array([1, -1*root]))
    
    E = np.real(E)
    
    return E, FnF, PnP


def Even_Odd_Parts(poly):
    '''
    Returns even and odd parts of a polynomial.
    Leading zeros are excluded.
    ''' 
    poly_e = np.copy(poly)
    poly_o = np.copy(poly)
    
    poly_e[-2::-2] = 0
    poly_o[-1::-2] = 0
      
    poly_e = np.trim_zeros(poly_e, 'f')
    poly_o = np.trim_zeros(poly_o, 'f')
    
    return poly_e, poly_o

        
def X1O(E, F):
    '''
    Forms the X1O (X of port 1(input), Open) polynomial
    For the case P(lambda) is even
    ''' 
    Ee, Eo = Even_Odd_Parts(E)
    Fe, Fo = Even_Odd_Parts(F)
    
    X1On = np.polysub(Ee, Fe)
    X1On = np.trim_zeros(X1On, 'f')
    
    X1Od = np.polyadd(Eo, Fo)
    X1Od = np.trim_zeros(X1Od, 'f')
    
    return Ee, Eo, Fe, Fo, X1On, X1Od
           

def X2O(E, F):
    '''
    Forms the X2O (X of port 2(output), Open) polynomial
    For the case P(lambda) is even
    ''' 
    Ee, Eo = Even_Odd_Parts(E)
    Fe, Fo = Even_Odd_Parts(F)
    
    X2On = np.polyadd(Ee, Fe)
    X2On = np.trim_zeros(X2On, 'f')
    
    X2Od = np.polyadd(Eo, Fo)
    X2Od = np.trim_zeros(X2Od, 'f')
    
    return Ee, Eo, Fe, Fo, X2On, X2Od                         
                                                  
                                                                                                    
def As_dB(n, p, theta_deg):
    '''
    See EQN(40) page 294 for rejection calculation.
    Delta_a for even or odd is used in EQN(40).
    
    n = number of sections or lowpass order 
    p = reflection coefficient or % of reflection
    theta_deg: theta in degrees such that sin(theta)=wc/ws. See page 292
    For even orders, theta_deg is not equal to wc/ws.
    '''
    _, delta_a, _ = a_v(n, theta_deg)
    rej_dB = 10*np.log10(1+1/((delta_a**4)*(1/p**2-1)))
    return rej_dB
    
    
def Plot_Poly(FnF, PnP, wc_p, rej_dB):
    #test plot of transfer function
      
    w = np.linspace(0.0001,10,num=100000)        
    lamb = 1j*w*wc_p 
    ws = 1/wc_p**2
    
    Kn2 = np.polyval(FnF,lamb)
    Kd2 = np.polyval(PnP,lamb) 
    K2 = np.real(Kn2)/np.real(Kd2)

    S21 = abs( (1/(1+K2)) )
    S21_dB = 10*np.log10(S21)
    
    ##As evaluated by plugging in ws.
    ##Not accurate due to the precision of number of points and steps.
    #ws = 1/(wc_p)**2
    #As = S21_dB[int(ws*10000)-1]
    
    #See EQN(40) page 294 for rejection
    As = -1* rej_dB
    S11 = abs( 1-S21 )
    S11_dB = 10*np.log10(S11)

    plt.clf()
    plt.plot(w, S21_dB, label = '$|S_{21}|^2$')
    plt.plot(w, S11_dB, label = '$|S_{11}|^2$')
    plt.legend(loc = 'lower right', shadow=False, fontsize = 'large')
    
    #Plot stop band
    plt.plot([ws, w[-1]], [As, As], 'g', linestyle=':')
    plt.plot([ws, ws], [As, 0], 'g', linestyle=':')
    txt = '('+ str(round(ws, 2)) + ', ' + str(round(As, 2)) + ')' 
    plt.text(round(ws,0) + 0.1, round(As,0)+2, txt, fontsize = 12)
     
    plt.axis([0, w[-1], np.floor(As/10)*10-20, 0])
    plt.xticks(np.arange(0, 11, 1))
    plt.grid()
    plt.show()

        
def Extract_Order(co_tz):
    '''
    forms an array of ordered transmission zero frequencies
    such that the extraction process will not yield negative components
    co_tz: coefficient from K_ representing inverse of normalized transmission
    zeros.
    tzop: array of normalized transmission zero frequencies that are ordere
    and padded with zeros so the indices match that of Saal and Ulbrich(SU)  
    '''
    tz = 1/(np.sort(co_tz))
    tzo = tz[::2]
    tzo = np.append(tzo, np.sort(tz[1::2]))
    
    #pad zeroes into tzo so the indices match SU
    tzop = np.array([0])
    for z in tzo:
        tzop = np.append(tzop,[0,z])
    
    return tzop
    
    
def Extract(n, p, tzop, E, F, wc_p, cauer_type = 'a'):
    cap_array = np.array([])
    ind_array = np.array([])
    omega_array = np.array([])
    
    _, _, _, _, X1On, X1Od = X1O(E, F)

    B_num = np.copy(X1Od)
    B_den = np.copy(X1On)

    #element extraction SU p306
    index=2
    while index < n:
        #shunt removal
        BdivL_num = B_num[:-1] #equivalent to B divided by lambda when constant term of B is 0
        BdivL_num_eval = np.polyval(BdivL_num, 1j*tzop[index])
        BdivL_den_eval = np.polyval(B_den, 1j*tzop[index])
        c_shnt = BdivL_num_eval/BdivL_den_eval
    
        #update component and frequency arrays
        cap_array = np.append(cap_array, np.real(c_shnt))
        ind_array = np.append(ind_array, [0])
        omega_array = np.append(omega_array, [0])       
    
        #update polynomila B
        Lc = np.array([np.real(c_shnt), 0])  #lambda*c(extracted)
        B_num = np.polysub(B_num, np.polymul(Lc, B_den))
        
        #series removal
        temp_n = np.polymul(np.array([1,0]), B_num)
        temp_n = np.polydiv(temp_n, np.array([1, 0, tzop[index]**2]))[0]
            
        temp_ne = np.polyval(temp_n, 1j*tzop[index])
        temp_de = np.polyval(B_den, 1j*tzop[index])
        c_srs = temp_ne/temp_de
        
        #update component and frequency arrays
        cap_array = np.append(cap_array, np.real(c_srs))
        w = tzop[index]
        omega_array = np.append(omega_array, [w])
        l_srs = 1/(w)**2/(np.real(c_srs))
        ind_array = np.append(ind_array, [l_srs])
    
        #update polynomial B
        temp1=np.polymul(B_den,np.array([1, 0, tzop[index]**2]))
        temp2=np.polymul(B_num,np.array([1/np.real(c_srs), 0]))
        B_den=np.polysub(temp1,temp2)
        B_num=np.polymul(B_num,np.array([1,0,tzop[index]**2]))
    
        index+=2

    #shunt removal
    BdivL_num = B_num[:-1] #B divided by lambda when constant term of B is 0
    BdivL_num_eval = BdivL_num[0]
    BdivL_den_eval = B_den[0]
    c_shnt = BdivL_num_eval/BdivL_den_eval

    #update component and frequency arrays
    cap_array = np.append(cap_array, np.real(c_shnt))
    ind_array = np.append(ind_array, [0])
    omega_array = np.append(omega_array, [0])

    #Lc = np.array([np.real(c_shnt), 0])  #lambda*c(extracted)
    #B_num = np.polysub(B_num, np.polymul(Lc, B_den))
    
    zout = 1
    
    #For even order filters, extract series L by using X2O
    if n%2 == 0:
        if cauer_type == 'b':
            zout = (1-p)/(1+p)
            
        #see table 4.4 page 129 in Zverev for X2O
        _, _, _, _, X2On, X2Od = X2O(E, F)
        l_srs = (X2On[0])/(X2Od[0])
        
        cap_array = np.append(cap_array, 0)
        ind_array = np.append(ind_array, l_srs*zout)
        omega_array = np.append(omega_array, [0])

    #Denormalize component values and frequencies
    cap_array = cap_array*wc_p
    ind_array = ind_array*wc_p
    omega_array = omega_array/wc_p
    
    #Pad zeros so indices match those of SU
    cap_array = np.insert(cap_array, 0, 0)
    cap_array = np.append(cap_array, 0)
    
    ind_array = np.insert(ind_array, 0, 0)
    ind_array = np.append(ind_array, 0)
    
    omega_array = np.insert(omega_array, 0, 0)
    omega_array = np.append(omega_array, 0)
    
    terms = np.array([1])
    terms = np.append(terms, np.zeros(n))
    terms = np.append(terms, zout)
    
    return cap_array, ind_array, omega_array, terms

def Eval_K(FnF, PnP, w, wc_p):
    #Outputs the S21 and S11 in dB given abs(K(lambda))^2
    #FnF: numerator of abs(K(lambda))^2
    #PnF: denominator of abs(K(lambda))^2
    #w: vector of omega for evaluation (rad)
    #wc_p: frequency normalization factor (rad)
           
    lamb = 1j*w*wc_p 
        
    Kn2 = np.polyval(FnF,lamb)
    Kd2 = np.polyval(PnP,lamb) 
    K2 = np.real(Kn2)/np.real(Kd2)

    S21 = abs( (1/(1+K2)) )
    S21_dB = 10*np.log10(S21)
    
    S11 = abs( 1-S21 )
    S11_dB = 10*np.log10(S11)
    
    return S11_dB, S21_dB
    
def Plot_S(S11_dB, S21_dB, w, ws, rej_dB):
    #test plot of transfer function 
    As = -1*rej_dB
    plt.clf()
    plt.plot(w, S21_dB, label = '$|S_{21}|^2$')
    plt.plot(w, S11_dB, label = '$|S_{11}|^2$')
    plt.legend(loc = 'lower right', shadow=False, fontsize = 'large')
    
    #Plot stop band
    plt.plot([ws, w[-1]], [As, As], 'g', linestyle=':')
    plt.plot([ws, ws], [As, 0], 'g', linestyle=':')
    txt = '('+ str(round(ws, 2)) + ', ' + str(round(As, 2)) + ')' 
    plt.text(ws + .01*w[-1], round(As,0)+1, txt, fontsize = 12)
     
    plt.axis([0, w[-1], np.floor(As/10)*10-20, 0])
    plt.xticks(np.arange(0, w[-1]+1, 1))
    plt.grid()
    plt.show()
