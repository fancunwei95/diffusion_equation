import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.linalg import solve_banded
from scipy.special import erf,erfc
plt.rc("text", usetex=True)

def ini_f(x):
    return 1.0

def res_f(x,t):
    a = erf(x/np.sqrt(t)/2.0)
    b = np.exp(x+t)
    c = erfc(np.sqrt(t)+x/np.sqrt(t)/2.0)
    return a+b*c

def initial(ini_f,x_min,x_max,N):
    x_list = np.linspace(x_min,x_max,N + 1)
    return x_list, ini_f(x_list)

##### explicit integration #################

def explicit(x_list,ini_u,M, t_coeff):
    dx = x_list[1]-x_list[0]
    u_list = np.zeros((M+1,len(x_list)))
    change_list = np.zeros((M+1,2))
    u_list[0] = ini_u
    t_list = np.zeros(M+1)
    count = 0
    dt = t_coeff*0.5*dx*dx
    alpha = dt/(dx*dx)
    change_list[0][0] = np.sum(u_list[0])*dx
    change_list[0][1] = np.sum(u_list[0])*dx
    while count < M :
        count = count + 1
        u_old = u_list[count-1]
        u_list[count][1:-1] = u_old[1:-1]+alpha*(u_old[2:]-2.0*u_old[1:-1]+u_old[:-2])
        #u_list[count][0] = u_list[count][1]/(1.0+dx)
        u_list[count][0] = u_list[count][1]/(2.0+dx)*(2.0-dx)
        u_list[count][-1] = 1.0
        change_list[count][0] = np.sum(u_list[count])*dx
        change_list[count][1] = change_list[count-1][1] - u_list[count][0]*dt
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list,change_list

def explicit_plot(t_list, x_list, u_list, change_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    
    plt.figure(1)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red", label = "analytical")
    plt.plot(x_list, u_list[m], color = "blue", label = "calculated")
    plt.suptitle("t = " +str(t_list[m]))

    plt.figure(2)
    dt = t_list[1] - t_list[0]
    #plt.plot(t_list[:-1], change_list[:-1,1], color = "red" , label ="analytical")
    #plt.plot(t_list[:-1], change_list[:-1,0], color = "blue", label ="calculated")
    plt.plot(t_list[:-1], (change_list[:-1,0]- change_list[:-1,1])/change_list[:-1,1], color = "blue", label ="percent error")
    plt.legend()
    
    plt.show()

def explicit_main():
    M = 50000
    N = 1000
    x_min = 0.0
    x_max = 20.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list, change_list = explicit(x_list, ini_u, M, t_coeff)
    explicit_plot(t_list, x_list, u_list, change_list, res_f, 50000)
    return 

def ex_conv_test(N_list, T ):
    x_min = 0.0
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
        t_list, x_list, u_list, change_list = explicit(x_list, ini_u, M, t_coeff)
        plt.plot(x_list, 2**i*(u_list[M]-res_f(x_list, t_list[M])), label = str(N) )
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
    plt.xlabel(r"$log(\Delta x)$", fontsize=24)
    plt.ylabel(r"$log(E)$", fontsize =24)
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
    dt = t_coeff*0.5*dx*dx
    alpha = dt/(dx*dx)
    matrix_band = np.zeros((3, mesh_num-2))
    matrix_band[1] = 2*alpha + 1.0
    matrix_band[0] = -alpha
    matrix_band[2] = -alpha
    matrix_band[0][0] = 0
    matrix_band[2][-1] = 0 
    matrix_band[1][0] = matrix_band[1][0] - alpha*(2.0-dx)/(2.0+dx)
    change_list = np.zeros((M+1,2))
    change_list[0][0] = np.sum(u_list[0])*dx
    change_list[0][1] = np.sum(u_list[0])*dx
    while count < M :
        count = count + 1
        u_old = deepcopy(u_list[count-1])
        #u_old_prime = u_old[1:-1] + alpha*(u_old[:-2]-2*u_old[1:-1]+u_old[2:])
        u_old_prime = u_old[1:-1]
        u_old_prime[-1] = u_old_prime[-1] + alpha
        u_list[count][1:-1] = solve_banded((1,1), matrix_band, u_old_prime)
        u_list[count][0] = u_list[count][1]/(2.0+dx)*(2.0-dx)
        u_list[count][-1] = 1.0
        change_list[count][0] = np.sum(u_list[count])*dx
        change_list[count][1] = change_list[count-1][1] - u_list[count][0]*dt
        t_list[count] = t_list[count-1]+dt
    return t_list,x_list,u_list, change_list   

def implicit_plot(t_list, x_list, u_list, change_list, res_f, m):
    m = min(u_list.shape[0]-1 , m)
    
    plt.figure(1)
    plt.plot(x_list, res_f(x_list, t_list[m]), color = "red")
    plt.plot(x_list, u_list[m], color = "blue")
    plt.suptitle("t = " +str(t_list[m]))
    
    plt.figure(2)
    plt.plot(x_list, res_f(x_list, t_list[m]) - u_list[m], color = "red")
    plt.suptitle("t = " +str(t_list[m]))

    plt.figure(3)
    #plt.plot(t_list[:-1], change_list[:-1,1], color = "red", label = "-du/dx*dt")
    #plt.plot(t_list[:-1], change_list[:-1,0], color = "blue", label ="calculated ")
    plt.plot(t_list[:-1], (change_list[:-1,0]- change_list[:-1,1])/change_list[:-1,1], color = "blue", label ="percent error")
    plt.show()

def implicit_main():
    M = 50000
    N = 400
    x_min = 0.0
    x_max = 40.0
    t_coeff = 0.2
    x_list, ini_u = initial(ini_f, x_min, x_max, N)
    t_list, x_list, u_list, change_list = implicit(x_list, ini_u, M, t_coeff)
    implicit_plot(t_list, x_list, u_list, change_list, res_f, 50000)
    return 

def im_conv_test(N_list, T ):
    x_min = 0.0
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
        t_list, x_list, u_list, change_list = implicit(x_list, ini_u, M, t_coeff)
        plt.plot(x_list, 2**i*(u_list[M]-res_f(x_list, t_list[M])), label = str(N) )
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
    plt.xlabel(r"$log(\Delta x)$", fontsize=24)
    plt.ylabel(r"$log(E)$", fontsize =24)
    plt.legend(loc="best")
    plt.show()


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
    number_list = deepcopy(bin_list[1:-1])
    if normalize:
        bin_list = bin_list/np.sum(bin_list[1:-1])/dx
    error_list = np.zeros(len(bin_list)-2) 
    for i in range(len(error_list)):
        if number_list[i] ==0:
            error_list[i] = 0.0 
            continue
        error_list[i] = np.sqrt((1-bin_list[i+1]) * bin_list[i+1] /number_list[i]  )
    return bin_list, error_list

def initialize(part_num,bin_num, x_min, x_max):
    bin_label = np.linspace(x_min,x_max, bin_num+1)
    part_list = np.zeros(part_num)
    bin_list,error_list = bin_part(bin_label, part_list)
    return bin_label, bin_list, part_list, error_list

def maximum(array, number):
    for i in range(len(array)):
        if array[i] < number:
            array[i] = number
    return array

def nonezero_stat(array):
    count = 0 
    mean = 0.0
    sqrt_mean = 0.0
    for i in range(len(array)):
        if array[i] < 0 :
            continue
        count += 1
        mean += array[i]
        sqrt_mean += array[i]*array[i]
    mean = mean/count
    sqrt_mean = sqrt_mean/count
    return mean, sqrt_mean

def monte_carlo(bin_label, bin_list, part_list, M, t_coeff):
    u_list = np.zeros((M+1,len(bin_label)-1))
    error = np.zeros((M+1, len(bin_label)-1))
    u_list[0] = bin_list[1:-1]
    t_list = np.zeros(M+1)
    dx = bin_label[1] - bin_label[0] 
    dt = t_coeff*0.5*dx*dx
    count = 0 
    part_num = len(part_list)
    u_der = (u_list[0][1]- u_list[0][0])/dx 
    u_0 = u_der + 1.0/dx
    factor = dx
    conv_list = np.zeros((M+1,2))
    conv_list[0][0] = np.sum(u_list[0])*dx
    conv_list[0][1] = np.sum(u_list[0])*dx
    while count < M:
        count = count + 1
        mean, sqrt_mean = nonezero_stat(part_list) 
        change_list = np.random.normal(0.0,1.0,part_num)*np.sqrt(abs(2.0 + u_der*sqrt_mean)*dt ) - (u_0+ u_der*mean)*dt
        #change_list = np.random.normal(0.0,1.0,part_num)*np.sqrt((2.0))
        part_list += change_list 
        bin_list, error_list = bin_part(bin_label, part_list)
        u_list[count] = bin_list[1:-1]
        error[count] = error_list
        u_der = (u_list[count][1] - u_list[count][0])/dx
        print count, factor, u_der, u_list[count][0]
        if u_list[count][0] == u_der:
            factor = 1.0
        else:
            factor = 1.0/(u_list[count][0] - u_der)
        u_0 = u_der + 1.0/factor
        u_list[count][0] = u_0
        u_list[count] = u_list[count] * factor
        error[count] = error[count]*factor
        t_list[count] = t_list[count-1] + dt
        conv_list[count][0] = np.sum(u_list[count])*dx
        conv_list[count][1] = conv_list[count-1][1] - u_list[count][0]*dt

    return t_list, u_list, error, conv_list

def mc_plot(t_list, bin_label, u_list, error,  conv_list,m):
    bin_label_c = 0.5* (bin_label[:-1] + bin_label[1:])
    plt.figure(1)
    plt.errorbar(bin_label_c, 1.0-u_list[m], yerr= error[m], color = "blue")
    plt.plot(bin_label_c, res_f(bin_label_c, t_list[m]), color = "red")
    plt.suptitle("t = "+str(t_list[m]))
    
    plt.figure(2)
    plt.plot(t_list[:-1], conv_list[:-1,1], color = "red", label = "-du/dx*dt")
    plt.plot(t_list[:-1], conv_list[:-1,0], color = "blue", label ="calculated ")
    
    plt.legend()
    plt.show()
    return

def mc_main():
    part_num = 10000
    bin_num = 200
    x_min = 0.0
    x_max = 20.0
    bin_label, bin_list, part_list, error = initialize(part_num, bin_num, x_min, x_max)
    t_list, u_list, error, conv_list= monte_carlo( bin_label, bin_list, part_list, 300, 0.2 )
    mc_plot(t_list, bin_label, u_list, error, conv_list,  300)

mc_main()
#explicit_main()
#implicit_main()
#ex_conv_test([100, 200, 400, 800, 1600, 3200],0.2)
#im_conv_test([100, 200, 400, 800, 1600, 3200], 0.2)
