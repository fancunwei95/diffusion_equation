import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.linalg import solve_banded
plt.rc("text", usetex=True)
def ini_f(x,dt):
    #return res_f(x,dt) 
    x = x -2.0
    return np.exp(-1.0*x*x/(0.0001))/(np.sqrt(np.pi*(0.0001)))

#def res_f(x_l,t):
#    #a = 1.0/np.sqrt(1.0+4.0*t)
#    r_l = np.zeros(len(x_l))
#    for i in range(len(x_l)):
#        x = x_l[i]
#        a = np.sqrt( 1.0/(2.0*np.pi*(1.0-np.exp(-2.0*t) )) )
#        b = -(x-2.0*np.exp(-t))**2
#        c = 2.0*(1.0-np.exp(-2.0*t))
#        r_l[i] = a*np.exp(b/c)
#    return r_l

def res_f(x,t):
    e = 2.0*(1.0-np.exp(-2.0*t))
    a = np.sqrt( 1.0/(e*np.pi)) 
    b = -(x-2.0*np.exp(-t))*(x-2.0*np.exp(-t))
    return a*np.exp(b/e)



def initial(ini_f,x_min,x_max,N):
    x_list = np.linspace(x_min,x_max,N + 1)
    dx = x_list[1] - x_list[0]
    dt = 0.2*0.5*dx*dx
    u_list = ini_f(x_list,dt)
    u_list = u_list/(np.sum(u_list[:-1]+u_list[1:])*0.5*dx)
    return x_list, ini_f(x_list,dt)

##### explicit integration #################

def explicit(x_list,ini_u,M, t_coeff):
    dx = x_list[1]-x_list[0]
    u_list = np.zeros((M+1,len(x_list)))
    u_list[0] = ini_u
    t_list = np.zeros(M+1)
    count = 0
    
    dt = t_coeff*0.5*dx*dx
    t_list[0] = dt
    alpha = dt/(dx*dx)
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        xu_list = x_list*u_old
        u_list[count][1:-1] = u_old[1:-1]+ (xu_list[2:]-xu_list[:-2])*dt/(2.0*dx) + alpha*(u_old[2:]-2.0*u_old[1:-1]+u_old[:-2])
        u_list[count][0] = 0.0
        u_list[count][-1] = 0.0
        normal = np.sum(0.5*(u_list[count][:-1] + u_list[count][1:])*dx)
        u_list[count] = u_list[count]/normal
        t_list[count] = t_list[count-1]+dt
    
    return t_list,x_list,u_list

def explicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    plt.show()

def explicit_plot_list(t_list, x_list, u_list, res_f, M):
    plt.figure()
    for m in M:
        m = min(u_list.shape[0]-1 , m)
        plt.plot(x_list, res_f(x_list, t_list[m]))
        plt.plot(x_list, u_list[m], ls = "--", label = "t = "+str(t_list[m]))
    plt.legend(loc = "best")
    plt.show()

def explicit_e_plot_list(t_list, x_list, u_list, res_f, M):
    plt.figure()
    for m in M:
        m = min(u_list.shape[0]-1 , m)
        plt.plot(x_list, u_list[m] - res_f(x_list, t_list[m]), label = "t = "+str(t_list[m]))
    plt.legend(loc = "best")
    plt.show()

def ex_conv_test(N_list, T ):
    x_min = -20.0
    x_max = 20.0
    t_coeff = 0.4
    error_list = np.zeros(len(N_list))
    plt.figure(1)
    for i in range(len(N_list)):
        N = N_list[i]
        dx = (x_max-x_min)/N
        dt = t_coeff*0.5*dx*dx
        M = int(round(T/dt))
        x_list, ini_u = initial(ini_f, x_min, x_max, N)
        t_list, x_list, u_list = explicit(x_list, ini_u, M, t_coeff)
        plt.plot(x_list, 4**i*(u_list[M]-res_f(x_list, t_list[M])), label = str(N) )
        #plt.plot(x_list, u_list[M], label = str(N) )
        #error_list[i] = np.linalg.norm(u_list[M]-res_f(x_list, t_list[M]))*dx
        error_list[i] = np.sum(np.abs(u_list[M]-res_f(x_list, t_list[M]))*dx)
    plt.legend(loc="best")
    plt.figure(2)
    dx_list= (x_max-x_min)/np.array(N_list)
    fit = np.polyfit(np.log(dx_list), np.log(error_list),1)
    m,b = fit
    fit_1d = np.poly1d(fit)
    plt.plot(np.log(dx_list), np.log(error_list),"o", label = "measured")
    plt.plot(np.log(dx_list), fit_1d(np.log(dx_list)), label = str(m)+"x"+"+"+ str(b))
    plt.legend(loc="best")
    plt.show()

def explicit_main():
    M = 320
    N = 200
    x_min = -20.0
    x_max = 20.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = explicit(x_list, ini_u, M, t_coeff)
    #explicit_plot(t_list, x_list, u_list,res_f, 1)
    explicit_plot_list(t_list, x_list, u_list, res_f, [1,20,40,160,320])
    explicit_e_plot_list(t_list, x_list, u_list, res_f, [1,20,40,160,320])
    return 

##### implicit integration ###################

def implicit(x_list,ini_u,M, t_coeff):
    dx = x_list[1]-x_list[0]
    mesh_num = len(x_list)
    u_list = np.zeros((M+1,mesh_num))
    u_list[0] = ini_u
    t_list = np.zeros(M+1)
    count = 0
    dt = t_coeff*0.5*dx*dx
    alpha = dt/(dx*dx)
    beta = dt/(2.0*dx)
    matrix_band = np.zeros((3, mesh_num-2))
    matrix_band[1] = 2*alpha + 1.0
    matrix_band[0][1:] = -alpha-beta*x_list[2:-1]
    matrix_band[2][:-1] = -alpha+beta*x_list[1:-2]
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        #u_old_prime = u_old[1:-1] + alpha*(u_old[:-2]-2*u_old[1:-1]+u_old[2:])
        u_old_prime = u_old[1:-1]
        u_list[count][1:-1] = solve_banded((1,1), matrix_band, u_old_prime)
        u_list[count][0] = 0.0
        u_list[count][-1] = 0.0
        normal = np.sum(0.5*(u_list[count][:-1] + u_list[count][1:])*dx)
        u_list[count] = u_list[count]/normal       
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list   

def implicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    plt.show()

def implicit_e_plot_list(t_list, x_list, u_list, res_f, M):
    plt.figure()
    for m in M:
        m = min(u_list.shape[0]-1 , m)
        plt.plot(x_list, u_list[m] - res_f(x_list, t_list[m]), label = "t = "+str(t_list[m]))
    plt.legend(loc = "best")
    plt.show()

def implicit_plot_list(t_list, x_list, u_list, res_f, M):
    for m in M:
        m = min(u_list.shape[0]-1 , m)
        plt.plot(x_list, res_f(x_list, t_list[m]))
        plt.plot(x_list, u_list[m], "o", label = "t = "+str(t_list[m]))
    plt.legend(loc = "best")
    plt.show()

def im_conv_test(N_list, T ):
    x_min = -20.0
    x_max = 20.0
    t_coeff = 0.4
    error_list = np.zeros(len(N_list))
    plt.figure(1)
    for i in range(len(N_list)):
        N = N_list[i]
        dx = (x_max-x_min)/N
        dt = t_coeff*0.5*dx*dx
        M = int(round(T/dt))
        x_list, ini_u = initial(ini_f, x_min, x_max, N)
        t_list, x_list, u_list = implicit(x_list, ini_u, M, t_coeff)
        plt.plot(x_list, 2**i*(u_list[M]-res_f(x_list, t_list[M])), label = str(N) )
        #plt.plot(x_list, u_list[M], label = str(N) )
        #error_list[i] = np.linalg.norm(u_list[M]-res_f(x_list, t_list[M]))*dx
        error_list[i] = np.sum(np.abs(u_list[M]-res_f(x_list, t_list[M])))*dx
    plt.legend(loc="best")
    plt.figure(2)
    dx_list= (x_max-x_min)/np.array(N_list)
    fit = np.polyfit(np.log(dx_list), np.log(error_list),1)
    m,b = fit
    fit_1d = np.poly1d(fit)
    plt.plot(np.log(dx_list), np.log(error_list),"o", label = "measured")
    plt.plot(np.log(dx_list), fit_1d(np.log(dx_list)), label = str(m)+"x"+"+"+ str(b))
    plt.legend(loc="best")
    plt.show()

def implicit_main():
    M = 1000
    N = 2000
    x_min = -50.0
    x_max = 50.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = implicit(x_list, ini_u, M, t_coeff)
    implicit_plot_list(t_list, x_list, u_list, res_f, [200,400,800])
    implicit_e_plot_list(t_list, x_list, u_list, res_f, [200,400,800])
    #implicit_plot(t_list, x_list, u_list,res_f, 100)
    return 

##### monte carlo simulation #################

def bin_part(bin_label, part_l, normalize = True):
    part_list = deepcopy(part_l)
    part_list.sort()
    bin_list = np.zeros(len(bin_label)+1)
    m = 0
    max_b = bin_label[-1]
    dx = bin_label[1] - bin_label[0]
    for n in range(len(part_list)):
        num = part_list[n]
        upper = bin_label[m]
        if (num >= max_b):
            bin_list[-1] += len(part_list) - n 
            break
        while (num >= upper):
            m = m + 1 
            upper = bin_label[m]
        bin_list[m] += 1
    assert len(part_list) == np.sum(bin_list)
    if normalize:
        bin_list = bin_list/len(part_list)/dx
    return bin_list

def initialize(part_num,bin_num, x_min, x_max):
    bin_label = np.linspace(x_min,x_max, bin_num+1)
    #part_list = np.random.normal(0.0, 1.0/np.sqrt(2.0), part_num)
    part_list = np.zeros(part_num)+2.0
    bin_list = bin_part(bin_label, part_list)
    return bin_label, bin_list, part_list

def monte_carlo(bin_label, bin_list, part_list, M, t_coeff):
    u_list = np.zeros((M+1,len(bin_label)-1))
    u_list[0] = bin_list[1:-1]
    t_list = np.zeros(M+1)
    dx = bin_label[1] - bin_label[0] 
    dt = t_coeff*0.5*dx*dx
    count = 0 
    part_num = len(part_list)
    while count < M:
        count = count + 1
        #DV = np.mean(part_list)
        part_list += -part_list*dt + np.random.normal(0.0,1.0,part_num)*np.sqrt(2.0*dt)
        #part_list += -DV*dt + np.random.normal(0.0,1.0,part_num)*np.sqrt(2.0*dt)
        u_list[count] = bin_part(bin_label, part_list)[1:-1]
        t_list[count] = t_list[count-1] + dt
    return t_list, u_list

def mc_plot(t_list, bin_label, u_list, m):
    bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
    plt.plot(bin_label_c, u_list[m], "o", color = "blue", label="mc")
    plt.plot(bin_label_c, res_f(bin_label_c, t_list[m]), color = "red", label="analytic")
    plt.suptitle("t = "+str(t_list[m]))
    plt.legend(loc="best")
    plt.show()
    return

def mc_main():
    part_num = 10000
    bin_num = 400
    x_min = -20.0
    x_max = 20.0
    bin_label, bin_list, part_list = initialize(part_num, bin_num, x_min, x_max)
    t_list, u_list = monte_carlo( bin_label, bin_list, part_list, 800, 0.2 )
    #t_list, u_list = discrete_mc( bin_label, bin_list, part_list, 20, 0.2 )
    bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for m in [50, 100, 200, 400, 800]:
        plt.plot(bin_label_c, u_list[m], "o--", label=str(t_list[m]))
        plt.plot(bin_label_c, res_f(bin_label_c, t_list[m]))
        print m
    plt.legend(loc="best")
    plt.xlabel("x", fontsize=32)
    plt.ylabel("f", fontsize=32)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    plt.show()
    

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for m in [50, 100, 200, 400, 800]:
        plt.plot(bin_label_c, u_list[m] -res_f(bin_label_c, t_list[m]), label=str(t_list[m]))
        print m
    plt.legend(loc="best")
    plt.xlabel("x", fontsize=32)
    plt.ylabel("error", fontsize=32)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    plt.show()
    #plt.suptitle("t = "+str(t_list[m]))
    #mc_plot(t_list, bin_label, u_list, 1000)

def mc_conv_test(N_list, T ):
    x_min = -20.0
    x_max = 20.0
    t_coeff = 0.8
    bin_num = 500
    error_list = np.zeros(len(N_list))
    fig_1 = plt.figure(1)
    ax = fig_1.add_subplot(1,1,1)
    for i in range(len(N_list)):
        N = N_list[i]
        dx = (x_max-x_min)/bin_num
        dt = t_coeff*0.5*dx*dx
        M = int(round(T/dt))
        bin_label, bin_list, part_list = initialize(N, bin_num, x_min, x_max)
        t_list, u_list = monte_carlo(bin_label, bin_list, part_list, M, t_coeff )
        print i, len(part_list), N,M
        bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
        plt.plot(bin_label_c, (u_list[M]-res_f(bin_label_c, t_list[M])), label = str(N) )
        error_list[i] = np.sum(np.abs(u_list[M]-res_f(bin_label_c, t_list[M])))*dx
    plt.legend(loc="best")
    plt.ylabel(r"$f - f_{analytic}$", fontsize=32)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    
    plt.figure(2)
    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1,1,1)
    fit = np.polyfit(np.log(N_list), np.log(error_list),1)
    m,b = fit
    fit_1d = np.poly1d(fit)
    plt.plot(np.log(N_list), np.log(error_list),"o", label = "measured")
    plt.plot(np.log(N_list), fit_1d(np.log(N_list)), label = str(m)+"x"+"+"+ str(b))
    plt.legend(loc="best")
    plt.xlabel(r"$\log{(N)}$", fontsize=32)
    plt.ylabel(r"$\log{(E)}$", fontsize=32)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    plt.show()

#mc_main()
#explicit_main()
#implicit_main()
#im_conv_test([400, 800, 1600, 3200],0.4)
#im_conv_test([40, 60, 80, 120,  160],0.4)
ex_conv_test([20,40, 80, 160],0.4)
#mc_conv_test([500, 2000, 8000, 32000], 4.0)
