import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.linalg import solve_banded

def ini_f(x):
    return np.exp(-1.0*x*x)

def res_f(x,t):
    a = 1.0/np.sqrt(1.0+4.0*t)
    return a*np.exp(-a*a*x*x)

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
    dt = t_coeff*0.5*dx*dx
    alpha = dt/(dx*dx)
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        u_list[count][1:-1] = u_old[1:-1]+alpha*(u_old[2:]-2.0*u_old[1:-1]+u_old[:-2])
        u_list[count][0] = 0.0
        u_list[count][-1] = 0.0
        t_list[count] = t_list[count-1]+dt
    
    return t_list,x_list,u_list

def explicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    plt.show()

def explicit_main():
    M = 1000
    N = 1000
    x_min = -50.0
    x_max = 50.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = explicit(x_list, ini_u, M, t_coeff)
    explicit_plot(t_list, x_list, u_list,res_f, 900)
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
    matrix_band = np.zeros((3, mesh_num-2))
    matrix_band[1] = 2*alpha + 1.0
    matrix_band[0] = -alpha
    matrix_band[2] = -alpha
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        #u_old_prime = u_old[1:-1] + alpha*(u_old[:-2]-2*u_old[1:-1]+u_old[2:])
        u_old_prime = u_old[1:-1]
        u_list[count][1:-1] = solve_banded((1,1), matrix_band, u_old_prime)
        u_list[count][0] = 0.0
        u_list[count][-1] = 0.0
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list   

def implicit_plot(t_list, x_list, u_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    plt.show()

def implicit_main():
    M = 1000
    N = 1000
    x_min = -50.0
    x_max = 50.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list = implicit(x_list, ini_u, M, t_coeff)
    implicit_plot(t_list, x_list, u_list,res_f, 100)
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
    part_list = np.random.normal(0.0, 1.0/np.sqrt(2.0), part_num)
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
        part_list += np.random.normal(0.0,1.0,part_num)*np.sqrt(2.0*dt)
        u_list[count] = bin_part(bin_label, part_list)[1:-1]
        t_list[count] = t_list[count-1] + dt
    return t_list, u_list*np.sqrt(np.pi)


def discrete_mc(bin_label, bin_list, part_list, M, t_coeff):
    u_list = np.zeros((M+1,len(bin_label)-1))
    u_list[0] = bin_list[1:-1]
    t_list = np.zeros(M+1)
    dx = bin_label[1] - bin_label[0] 
    dt = 1.0
    count = 0 
    part_num = len(part_list)
    while count < M:
        count = count + 1
        part_list += np.sign(np.random.uniform(-0.5,0.5,part_num))
        u_list[count] = bin_part(bin_label, part_list)[1:-1]
        t_list[count] = t_list[count-1] + dt
    return t_list, u_list*np.sqrt(np.pi)

def mc_plot(t_list, bin_label, u_list, m):
    bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
    plt.plot(bin_label_c, u_list[m], color = "blue")
    plt.plot(bin_label_c, res_f(bin_label_c, t_list[m]), color = "red")
    plt.suptitle("t = "+str(t_list[m]))
    plt.show()
    return

def mc_main():
    part_num = 10000
    bin_num = 100
    x_min = -50.0
    x_max = 50.0
    bin_label, bin_list, part_list = initialize(part_num, bin_num, x_min, x_max)
    #t_list, u_list = monte_carlo( bin_label, bin_list, part_list, 500, 0.2 )
    t_list, u_list = discrete_mc( bin_label, bin_list, part_list, 20, 0.2 )
    time_array = [0,1,2,3,6,10,15]
    for i in time_array:
        mc_plot(t_list, bin_label, u_list, i)

mc_main()
#explicit_main()
#implicit_main()
