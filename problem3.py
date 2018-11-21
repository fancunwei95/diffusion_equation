import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.linalg import solve_banded
plt.rc("text", usetex=True)

def ini_f(x):
    return x

def res_f(x,t):
    return np.sign(x) * abs(x)**(1.0/3.0)

def initial(ini_f,x_min,x_max,N):
    x_list = np.linspace(x_min,x_max,N + 1)
    return x_list, ini_f(x_list)

##### explicit integration #################

def explicit(x_list,ini_u,M, t_coeff):
    dx = x_list[1]-x_list[0]
    u_list = np.zeros((M+1,len(x_list)))
    u_list[0] = ini_u
    t_list = np.zeros(M+1)
    count = 0
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        dt = np.nanmin(t_coeff*0.5*dx*dx/(3.0*u_old*u_old))
        alpha = alpha = dt/(dx*dx)
        u_old_p = u_old*u_old*u_old
        u_list[count][1:-1] = u_old[1:-1]+alpha*(u_old_p[2:]-2.0*u_old_p[1:-1]+u_old_p[:-2])
        u_list[count][0] = -1.0
        u_list[count][-1] = 1.0
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list

def explicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.scatter(x_list, u_list[m], color = "blue")
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

def explicit_main():
    M = 50000
    N = 100
    x_min = -1.0
    x_max = 1.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = explicit(x_list, ini_u, M, t_coeff)
    explicit_plot_list(t_list, x_list, u_list, res_f, [0,500,1000, 5000, 10000, 50000  ])
    #explicit_plot(t_list, x_list, u_list,res_f, 49000)
    return 

def ex_conv_test(N_list, T ):
    x_min = -1.0
    x_max = 1.0
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
        plt.plot(x_list, 2**i*(u_list[M]-res_f(x_list, t_list[M])), label = str(N) )
        #error_list[i] = np.linalg.norm(u_list[M]-res_f(x_list, t_list[M]))*dx
        error_list[i] = np.sum(np.abs(u_list[M]-res_f(x_list, t_list[M]))*dx)/N
        print str(t_list[M])
        print i
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

##### implicit integration ###################

def implicit(x_list,ini_u,M, t_coeff):
    dx = x_list[1]-x_list[0]
    mesh_num = len(x_list)
    u_list = np.zeros((M+1,mesh_num))
    u_list[0] = ini_u
    t_list = np.zeros(M+1)
    count = 0
    theta = 1.0
    dt = t_coeff*0.5*dx*dx
    alpha = dt/(dx*dx)
    A = 3.0* theta * alpha
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        u_old_2 = u_old*u_old
        u_old_3 = u_old_2*u_old
        
        matrix_band = np.zeros((3, mesh_num-2))
        matrix_band[1] = (2.0*A+1.0)*u_old_2[1:-1]
        matrix_band[0][1:] = -A*u_old_2[2:-1]
        matrix_band[2][:-1] = -A*u_old_2[1:-2]

        u_old_prime = alpha*(u_old_3[2:] - 2.0*u_old_3[1:-1] + u_old_3[:-2])
        w_list = solve_banded((1,1), matrix_band, u_old_prime)
        u_list[count][1:-1] = w_list + u_old[1:-1]
        u_list[count][0] = -1.0
        u_list[count][-1] = 1.0
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list   

def implicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    plt.show()

def implicit_plot_list(t_list, x_list, u_list, res_f, M):
    plt.figure()
    for m in M:
        m = min(u_list.shape[0]-1 , m)
        plt.plot(x_list, res_f(x_list, t_list[m]))
        plt.plot(x_list, u_list[m], ls = "--", label = "t = "+str(t_list[m]))
    plt.legend(loc = "best")
    plt.show()

def implicit_main():
    M = 50000
    N = 999
    x_min = -1.0
    x_max = 1.0
    t_coeff = 5.0
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = implicit(x_list, ini_u, M, t_coeff)
    implicit_plot_list(t_list, x_list, u_list, res_f, [0,500,1000, 5000, 10000, 50000  ])
    return 

def im_conv_test(N_list, T ):
    x_min = -1.0
    x_max = 1.0
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
        #error_list[i] = np.linalg.norm(u_list[M]-res_f(x_list, t_list[M]))*dx
        error_list[i] = np.sum(np.abs(u_list[M]-res_f(x_list, t_list[M]))*dx)
        print str(t_list[M])
        print i
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


##### monte carlo simulation #################

def bin_part(bin_label, part_list, normalize = True):
    M = len(bin_label) 
    N = len(part_list)
    bin_list = np.zeros(M+1)
    min_b = bin_label[0]
    max_b = bin_label[-1]
    dx = bin_label[1] - bin_label[0]
    for n in range(len(part_list)):
        num = part_list[n][0]
        value = part_list[n][1]
        m = int(np.ceil((num-min_b)/dx))
        m = max(0,m)
        m = min(M,m)
        bin_list[m] += value   
        part_list[n][2] = m
    assert 0 == np.sum(bin_list)
    if normalize:
        bin_list = bin_list/N/dx
    factor = (1.0/bin_list[-2] -1.0/bin_list[1])*0.5
    index_list = np.zeros(N)
    D_array = 3.0*(bin_list*bin_list)
    D_array[0] = D_array[1]
    D_array[-1] = D_array[-2]
    for i in range(N):
        index_list[i] = D_array[part_list[i][2]]
    return bin_list*factor, index_list, part_list

def initialize(part_num,bin_num, x_min, x_max):
    bin_label = np.linspace(x_min,x_max, bin_num+1)
    part_list_p = np.sqrt(np.random.uniform(0.0, 1.0, int(part_num/2.0)))
    part_list_n = -np.sqrt(np.random.uniform(0.0,1.0, int(part_num/2.0)))
    part_list = np.zeros((part_num,3))
    part_list[:part_num/2 ,0] = part_list_n
    part_list[:part_num/2 ,1] = -1.0
    part_list[part_num/2: ,0] = part_list_p
    part_list[part_num/2: ,1] = 1.0
    bin_list, D_list, part_list = bin_part(bin_label, part_list)
    return bin_label, bin_list, part_list, D_list

def monte_carlo(bin_label, bin_list, part_list, D_list, M, t_coeff):
    u_list = np.zeros((M+1,len(bin_label)-1))
    u_list[0] = bin_list[1:-1]
    t_list = np.zeros(M+1)
    dx = bin_label[1] - bin_label[0] 
    dt = t_coeff*0.5*dx*dx/np.nanmax(D_list)
    count = 0 
    while count < M:
        part_num = len(part_list)
        count = count + 1
        part_list[:,0] += np.random.normal(0.0,1.0,part_num)*np.sqrt(2.0*D_list*dt)
        bin_list, D_list, part_list = bin_part(bin_label, part_list)
        part_list[:,0] = maximum(part_list[:,0], bin_label[0]-dx*0.5)
        part_list[:,0] = minimum(part_list[:,0], bin_label[-1]+dx*0.5)
        u_list[count] = bin_list[1:-1]
        t_list[count] = t_list[count-1] + dt
        print count, part_num
    return t_list, u_list

def mc_plot(t_list, bin_label, u_list, m):
    bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
    plt.plot(bin_label_c, u_list[m], color = "blue")
    plt.plot(bin_label_c, res_f(bin_label_c, t_list[m]), color = "red")
    #plt.suptitle("t = "+str(t_list[m]))
    plt.show()
    return

def mc_main():
    part_num = 10000
    bin_num = 200
    x_min = -1.0
    x_max = 1.0
    bin_label, bin_list, part_list, D_list = initialize(part_num, bin_num, x_min, x_max)
    print bin_list
    t_list, u_list = monte_carlo( bin_label, bin_list, part_list, D_list,  10000, 0.8 )
    mc_plot(t_list, bin_label, u_list, 10000)

def maximum(array, number):
    for i in range(len(array)):
        if array[i] < number:
            array[i] = number
    return array

def minimum(array, number):
    for i in range(len(array)):
        if array[i] > number:
            array[i] = number
    return array

mc_main()
#explicit_main()
#implicit_main()
#ex_conv_test([20, 40, 80, 160], 4.0)
#im_conv_test([101, 201, 401, 801], 1.0)

