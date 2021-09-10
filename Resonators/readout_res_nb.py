import res_model as res
import numpy as np
import matplotlib.pyplot as plt


def plot_cable(n):
    #cable params
    fs = np.arange(500, 1000)#A frequency comb from 500 MHz to 1 GHz
    A0 = 0.95
    A1 = -0.03/500
    tau = 1/50
    phi = 0
    
    #the cable!
    general_cable = res.general_cable(f = fs, 
                                  delay = tau, 
                                  phi = phi,
                                  f_min = fs[0],
                                  A_mag = A0, 
                                  A_slope = A1/A0)
    
    #The plot
    plt.figure(figsize=(6,6))
    plt.scatter(np.real(general_cable)[0:n], 
            np.imag(general_cable)[0:n], c=fs[0:n])
    plt.vlines(0, -10, 10,linestyles='dotted')
    plt.hlines(0, -10, 10,linestyles='dotted')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Real($S_{21}$)")
    plt.ylabel("Imag($S_{21}$)")
    plt.show()
    return
    
def plot_IQ_Amp(f, S, n, markers = False, ghost_res = [False]):
    '''Function to simultaneously scrub through the amplitude and complex plots
    based on an index
    '''
    fig, (magax, compax) = plt.subplots(2,
                                    figsize = (8,12), 
                                    gridspec_kw=
                                    {'height_ratios': [1, 3]})
    
    magax.plot(f, np.abs(S))
    magax.plot(f[n], np.abs(S[n]), 'r*')
    magax.vlines(f[n], -1, 1.25, linestyles = 'dotted')
    magax.set_xlabel('Freq (MHz)')
    magax.set_ylabel('$|S_{21}|$')
    magax.set_ylim(0, 1.1)
    
    if ghost_res[0]:
        compax.plot(np.real(ghost_res), np.imag(ghost_res), 'k:', 
                    markersize = 0.7, lw = 0.7, markevery=2, 
                    label = 'Ideal Resonator')
        compax.legend(loc=0)
    
    compax.plot(np.real(S), np.imag(S))
    compax.plot(np.real(S[n]), np.imag(S[n]), 'r*')
    compax.set_ylabel('Imag($S_{21}$)')
    compax.set_xlabel('Real($S_{21}$)')
    compax.set_ylim(-1,1)
    compax.set_xlim(-1,1)
    compax.vlines(0, -1.5, 1.5,linestyles='dotted')
    compax.hlines(0, -1.5, 1.5,linestyles='dotted')
    
    if markers:
        magax.plot(f[n-markers], np.abs(S[n-markers]), 'm*')
        magax.plot(f[n+markers], np.abs(S[n+markers]), 'y*')
        compax.plot(np.real(S[n-markers]), np.imag(S[n-markers]), 'm*')
        compax.plot(np.real(S[n+markers]), np.imag(S[n+markers]), 'y*')
        
    plt.show()
    
    return
    
def plot_res_cable(freqs, f0, Q, Qe_real, Qe_theta, A0, A1, tau, phi, m = False):
    '''Function to look at complex and amplitude S21 while varying cable 
    or resonator'''

    Qe= Qe_real*(1 + 1j*np.tan(np.deg2rad(Qe_theta)))
   
    br = f0/(2*Q)
    br_i = np.int(br/(freqs[1]-freqs[0]))
   
    if m:
        br_i = False
        pass
    else:
        m = np.where(np.abs(freqs-f0) ==  np.min(np.abs(freqs-f0)))[0][0]
    
    phi0 = phi - 2*np.pi*(freqs[0]-f0)*tau
    
    S = res.resonator_cable(freqs, f0, Q, Qe_real, np.imag(Qe), 
                            delay= tau, 
                            phi = phi0, 
                            f_min = freqs[0], 
                            A_mag = A0, 
                            A_slope = A1/A0)
    
    ghost = res.resonator_cable(freqs, f0, Q, Qe_real, 0, 
                            delay=0, phi=0,f_min=freqs[0],A_mag=1, A_slope=0)
    
    plot_IQ_Amp(freqs, S, m, markers = br_i, ghost_res = ghost)
    return


def plot_comp_amp_Qe(f, S, n, markers = False, ghost_res = [False], new_min = False):
    '''Function to simultaneously scrub through the amplitude and complex plots
    based on an index
    '''
    fig, (magax, compax) = plt.subplots(2,
                                    figsize = (8,12), 
                                    gridspec_kw=
                                    {'height_ratios': [1, 3]})
    
    magax.plot(f, np.abs(S))
    magax.plot(f[n], np.abs(S[n]), 'r*')
    magax.set_xlabel('Freq (MHz)')
    magax.set_ylabel('$|S_{21}|$')
    magax.set_ylim(0, 1.3)
    magax.set_xlim(f[0], f[-1])
    
    compax.plot(np.real(S), np.imag(S))
    compax.plot(np.real(S[n]), np.imag(S[n]), 'r*')
    compax.set_ylabel('Imag($S_{21}$)')
    compax.set_xlabel('Real($S_{21}$)')
    compax.set_xlim(-0,1.5)
    compax.set_ylim(-0.75,0.75)
    compax.vlines(1, -1.5, 1.5,linestyles='dotted')
    compax.hlines(0, -1.5, 1.5,linestyles='dotted')
    
    if markers:
        magax.plot(f[n-markers], np.abs(S[n-markers]), 'm*')
        magax.plot(f[n+markers], np.abs(S[n+markers]), 'y*')
        compax.plot(np.real(S[n-markers]), np.imag(S[n-markers]), 'm*')
        compax.plot(np.real(S[n+markers]), np.imag(S[n+markers]), 'y*')
    
    if new_min:
        magax.plot(f[new_min], np.abs(S[new_min]), 'cX', label = "New Minima")
        magax.vlines([f[n],f[new_min]], -1, 1.25, linestyles = 'dotted')
        magax.hlines([1, np.abs(S[new_min])], 545, 555, linestyles = 'dotted')
        compax.plot(np.real(S[new_min]), np.imag(S[new_min]), 'cX', label = "New Minima")
    else:
        magax.vlines(f[n], -1, 1.25, linestyles = 'dotted')
        magax.hlines(1, 545, 555, linestyles = 'dotted')
    
    if ghost_res[0]:
        compax.plot(np.real(ghost_res), np.imag(ghost_res), 'k:', 
                    markersize = 0.7, lw = 0.7, markevery=2, 
                    label = 'Ideal Resonator')
        compax.legend(loc=0)
    
    plt.show()
    
    return

def plot_Qe(freqs, f0, Q, Qe_real, Qe_theta, A0, A1, tau, phi, m = False):
    '''Function to look at complex and amplitude S21 while varying cable 
    or resonator'''

    Qe= Qe_real*(1 + 1j*np.tan(np.deg2rad(Qe_theta)))
    
    br = f0/(2*Q)
    br_i = np.int(br/(freqs[1]-freqs[0]))
   
    if m:
        br_i = False
        pass
    else:
        m = np.where(np.abs(freqs-f0) ==  np.min(np.abs(freqs-f0)))[0][0]
    
    
    phi0 = phi - 2*np.pi*(freqs[0]-f0)*tau
    
    S = res.resonator_cable(freqs, f0, Q, Qe_real, np.imag(Qe), 
                            delay= tau, 
                            phi = phi0, 
                            f_min = freqs[0], 
                            A_mag = A0, 
                            A_slope = A1/A0)
    
    
    nearest_idx = np.where(np.abs(S) == np.min(np.abs(S)))[0][0]
    
    ghost = res.resonator_cable(freqs, f0, Q, Qe_real, 0, 
                            delay=0, phi=0,f_min=freqs[0],A_mag=1, A_slope=0)
    
    plot_comp_amp_Qe(freqs, S, m, markers = br_i, ghost_res = ghost, 
                     new_min = nearest_idx)
    return

def plot_amp_phase(f, S, n, markers = False):
    '''Function to simultaneously scrub through the amplitude and complex plots
    based on an index
    '''
    fig, (magax, phax) = plt.subplots(2,
                                    figsize = (8,6), 
                                    gridspec_kw=
                                    {'height_ratios': [1, 1]})
    
    magax.plot(f, np.abs(S))
    magax.plot(f[n], np.abs(S[n]), 'r*')
    magax.vlines(f[n], -1, 1.25, linestyles = 'dotted')
    magax.set_xlabel('Freq (MHz)')
    magax.set_ylabel('$|S_{21}|$')
    magax.set_ylim(0, 1.1)
    
    phax.plot(f, np.angle(S))
    phax.plot(f[n], np.angle(S[n]), 'r*')
    phax.vlines(f[n], -2, 2, linestyles = 'dotted')
    phax.set_xlabel('Freq (MHz)')
    phax.set_ylabel('$S_{21}$ Phase')
    phax.set_ylim(np.min(np.angle(S)), np.max(np.angle(S)))
    
    if markers:
        magax.plot(f[n-markers], np.abs(S[n-markers]), 'm*')
        magax.plot(f[n+markers], np.abs(S[n+markers]), 'y*')
        phax.plot(f[n-markers], np.angle(S[n-markers]), 'm*')
        phax.plot(f[n+markers], np.angle(S[n+markers]), 'y*')
        
    plt.show()
    
    return

def plot_res_phase(freqs, f0, Q, Qe_real, Qe_theta, A0, A1, tau, phi, m = False):
    '''Function to look at complex and amplitude S21 while varying cable 
    or resonator'''

    Qe= Qe_real*(1 + 1j*np.tan(np.deg2rad(Qe_theta)))
   
    br = f0/(2*Q)
    br_i = np.int(br/(freqs[1]-freqs[0]))
   
    if m:
        br_i = False
        pass
    else:
        m = np.where(np.abs(freqs-f0) ==  np.min(np.abs(freqs-f0)))[0][0]
    
    phi0 = phi - 2*np.pi*(freqs[0]-f0)*tau
    
    S = res.resonator_cable(freqs, f0, Q, Qe_real, np.imag(Qe), 
                            delay= tau, 
                            phi = phi0, 
                            f_min = freqs[0], 
                            A_mag = A0, 
                            A_slope = A1/A0)

    plot_amp_phase(freqs, S, m, markers = br_i)
    return


def mega_res_plot(f, S, n, markers = False, ghost_res = [False]):
    '''Function to simultaneously scrub through the amplitude and complex plots
    based on an index
    '''
    
    fig = plt.figure(figsize = (7,9))
    magax = fig.add_axes([0.05, 0.85, 0.4, 0.25])
    phax = fig.add_axes([0.55, 0.85, 0.4, 0.25])
    compax = fig.add_axes([0.05, 0.05, .9, 0.7])

    magax.plot(f, np.abs(S))
    magax.plot(f[n], np.abs(S[n]), 'r*')
    magax.vlines(f[n], -1, 1.25, linestyles = 'dotted')
    magax.set_xlabel('Freq (MHz)')
    magax.set_ylabel('$|S_{21}|$')
    magax.set_ylim(0, 1.1)
    
    phax.plot(f, np.angle(S))
    phax.plot(f[n], np.angle(S[n]), 'r*')
    phax.vlines(f[n], -2, 2, linestyles = 'dotted')
    phax.set_xlabel('Freq (MHz)')
    phax.set_ylabel('$S_{21}$ Phase')
    phax.set_ylim(np.min(np.angle(S)), np.max(np.angle(S)))
    phax.set_xlim(f[0], f[-1])
    phax.hlines(np.arange(-4, 4, 0.5), 545, 555, linestyles='dotted')
    
    if ghost_res[0]:
        compax.plot(np.real(ghost_res), np.imag(ghost_res), 'k:', 
                    markersize = 0.7, lw = 0.7, markevery=2, 
                    label = 'Ideal Resonator')
        compax.legend(loc=0)
    
    compax.plot(np.real(S), np.imag(S))
    compax.plot(np.real(S[n]), np.imag(S[n]), 'r*')
    compax.set_ylabel('Imag($S_{21}$)')
    compax.set_xlabel('Real($S_{21}$)')
    compax.set_ylim(-1,1)
    compax.set_xlim(-1,1)
    compax.vlines(0, -1.5, 1.5,linestyles='dotted')
    compax.hlines(0, -1.5, 1.5,linestyles='dotted')
    
    if markers:
        magax.plot(f[n-markers], np.abs(S[n-markers]), 'm*')
        magax.plot(f[n+markers], np.abs(S[n+markers]), 'y*')
        phax.plot(f[n-markers], np.angle(S[n-markers]), 'm*')
        phax.plot(f[n+markers], np.angle(S[n+markers]), 'y*')
        compax.plot(np.real(S[n-markers]), np.imag(S[n-markers]), 'm*')
        compax.plot(np.real(S[n+markers]), np.imag(S[n+markers]), 'y*')
        
    plt.show()
    
    return

def full_res_look(freqs, f0, Q, ratio_Qc_Qi_exp, Qe_theta, A0, A1, tau, phi, m = False):
    '''Function to look at complex, phase, and amplitude S21 while varying cable 
    or resonator'''
    
    ratio_Qc_Qi = 10**ratio_Qc_Qi_exp# = Q_c/Q_i = real Qe/Qi
    
    Qe_real = Q*(1 + ratio_Qc_Qi)

    Qe= Qe_real*(1 + 1j*np.tan(np.deg2rad(Qe_theta)))
   
    br = f0/(2*Q)
    br_i = np.int(br/(freqs[1]-freqs[0]))
   
    if m:
        br_i = False
        pass
    else:
        m = np.where(np.abs(freqs-f0) ==  np.min(np.abs(freqs-f0)))[0][0]
    
    phi0 = phi - 2*np.pi*(freqs[0]-f0)*tau
    
    S = res.resonator_cable(freqs, f0, Q, Qe_real, np.imag(Qe), 
                            delay= tau, 
                            phi = phi0, 
                            f_min = freqs[0], 
                            A_mag = A0, 
                            A_slope = A1/A0)
    
    ghost = res.resonator_cable(freqs, f0, Q, Qe_real, 0, 
                            delay=0, phi=0,f_min=freqs[0],A_mag=1, A_slope=0)
    
    mega_res_plot(freqs, S, m, markers = br_i, ghost_res = ghost)
    return

def Q_total(qi, r, delta):
    '''Function to make Q_total paramatrized by internal Q (Q_i),
    the ratio of r = Q_i/Q_c, and the fractional shift in Q_i 
    (i.e. Qi--> Qi(1-delta))
    '''
    Q = r*(1-delta)*qi/(r+1-delta)
    return Q

def S21(f, Qi, Qc, eps, f0):
    #A function for a resonator that includes the epsilon term rather
    #than Qe
    Qr = (Qi**-1 + Qc**-1)**-1
    dx = f/f0 -1
    QQ = Qr/Qc
    epQ = (1+1j*eps*QQ)
    
    first = (1+1j*eps)/epQ 
    second = 1 + 2j*Qr*dx/epQ

    return 1 - QQ*first/second

def plot_Q_shifts(r_exp):
    ## Plot parameters
    dmin, dmax, n = 0.0, 0.5, 25
    r = 10**r_exp
    Q_i =3.0e4*(r+1)/r
    fixQ = 3.0e4

    ## Plot the figure
    delta = np.linspace(dmin, dmax, n)
    Q = Q_total(Q_i, r, delta)
    plt.figure(figsize=(8,6))
    plt.plot(1-delta, Q)

    ## Set up the figure axes, etc.
    plt.title(f"Total Q: {fixQ}, $Q_i$:{Q_i}")
    plt.xlim(1-dmax, 1-dmin)
    plt.ylim(1.5e4,3.0e4)
    plt.xlabel('1-delta')
    plt.ylabel('Q')
    return

def plot_Q_shifts_S(r_exp):
    ## Plot parameters
    dmin, dmax, n = 0.0, 0.25, 6
    r = 10**r_exp
    Qi =3.0e4*(r+1)/r
    Qc = 3.0e4*(r+1)
    fixQ = 3.0e4
    #
    f0 =5.5e8
    br = f0/(2*fixQ)
    freqs = np.arange(f0 - 3*br, f0 + 3*br, 6*br/200)
    eps = 0.001

    ## Plot the figure
    delta = np.linspace(dmin, dmax, n)
    Q = Q_total(Qi, r, delta)
    plt.figure(figsize=(8,6))
    for d in delta:
        qi = Qi*(1-d)
        s = S21(freqs, qi, Qc, eps, f0)
        plt.plot(freqs, np.abs(s), label = np.round(d, 2))

    ## Set up the figure axes, etc.
    plt.title(f"Total Q: {fixQ}, $Q_i$:{Qi}")
    plt.xlabel('freqs')
    plt.legend(title="$\delta Q_i/Q_i$",loc=0)
    return

def params_in_to_ex(L, R, Cr, Cc):
    #Function to turn intrinsic circuit parameters into measurables
    z0 = 50
    C = Cr + Cc
    om0 = (np.sqrt(L*(Cr+Cc)))**-1
    f0 = om0/(2*np.pi)
    Qi = om0*L/R
    Qc = 2*C/(om0*z0*Cc**2)
    eps = Cr/(Qi*Cc)
    return f0, Qi, Qc, eps

def params_ex_to_in(f0, Qi, Qc, eps):
    #convert measured parameters to internal circuit params 
    Z0=50
    om0 = 2*np.pi*f0
    R = Z0*Qc/(2*Qi*(1+Qi*eps)**2)
    L = Z0*Qc/(2*om0*(1+Qi*eps)**2)
    Cc = 2*(1+Qi*eps)/(om0*Z0*Qc)
    Cr = 2*Qi*eps*(1+Qi*eps)/(om0*Z0*Qc)
    return L, R, Cr, Cc

def shift_LR_plot(del_L, del_R, IQ = False):
    '''A function to explore how the resonator changes when 
    you adjust both L and R'''
    #Initial params
    Qci = 4e4
    Qii = 1e5
    epsi = 0.001
    f0i = 5.5e8
    Qri = (Qii**-1 + Qci**-1)**-1
    br = f0i/(2*Qri)
    freqs = np.arange(f0i - 3*br, f0i + 3*br, 6*br/200)
    
    #Set initial params
    L0, R0, Cr, Cc = params_ex_to_in(f0i, Qii, Qci, epsi)
    
    #Adjust L & create new params, s21
    R = R0*(1+del_R)
    L = L0*(1+del_L)
    f0, Qi, Qc, eps = params_in_to_ex(L, R, Cr, Cc)
    s = S21(freqs, Qi, Qc, eps, f0)
    
    #plot the stuff
    plt.figure(figsize=(8,6))
    
    if IQ=="IQ":
        plt.plot(np.real(s), np.imag(s))
        s0 = S21(f0i,Qi, Qc, eps, f0)
        plt.plot(np.real(s0), -np.imag(s0), 'r*', label="Original $f_0$")
        s0 = S21(f0,Qi, Qc, eps, f0)
        plt.plot(np.real(s0), -np.imag(s0), 'b*', label="New $f_0$")
        plt.title(f"Resonator while varying L & R")
        plt.xlim(0,1.5)
        plt.ylim(-.75, .75)
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.legend(loc=0)
    else:
        plt.plot(freqs*10**-6, np.abs(s))
        plt.vlines(f0i*10**-6, 0,1, linestyles='dashed')
        plt.ylim(.2, .95)
        plt.title(f"Resonator while varying L & R")
        plt.xlabel("Freq (MHz)")
    return
