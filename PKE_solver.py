import numpy as np
import scipy
from scipy.integrate import solve_ivp
from numpy.linalg import solve
from scipy import integrate

'''
Point kinetics equations solver
Wei Xiao
Shanghai Jiao Tong University
School of Nuclear Science and Engineering
bearsanxw@gmail.com
2021-4-10
1. Raudu-IIA
2. Implicit-Euler
3. First-order SCM
4. Zero-order SCM
5. Second-order SCM
'''
class solver_PKE:
    def __init__(self,delay_group_num,time_start,time_end):
        self.delay_group_num = delay_group_num
        self.time_start = time_start
        self.time_end = time_end
    
    # Initialize parameters
    def initial_dynamics_parameters(self,neutron_life,delay_beta,delay_lambda):
        # Dynamics parameters
        # Average neutron generation time (scalar)
        self.neutron_life = neutron_life
        # The share of the delay neutrons (delay_group_num x 1 array)
        self.delay_beta = delay_beta
        self.total_beta = delay_beta.sum()
        # The decay constant of the precursors (delay_group_num x 1 array)
        self.delay_lambda = delay_lambda
    
    def initial_density(self,neutron_density,precursor_conc):
        # Initialize the neutron density and the precursor concentration
        self.neutron_density_init = neutron_density
        self.precursor_conc_init = precursor_conc

    def time_variant_reactivity(self,init_rho,end_rho,variant_type='linear'):
        self.init_rho = init_rho
        self.end_rho = end_rho
        self.variant_type = variant_type

    def initial_frequency(self,omega):
        self.init_omega = omega

    # Reactivity function
    def rho_func(self,t):
        if self.variant_type=='linear':
            rho = self.init_rho+(t-self.time_start)*(self.end_rho-self.init_rho)/(self.time_end-self.time_start)
        return rho
    
    def __define_H_matrix(self):
        group = self.delay_group_num
        rho_init = self.init_rho
        Lambda = self.neutron_life
        lamb = self.delay_lambda
        beta_sum = self.total_beta
        beta = self.delay_beta

        H = np.zeros((group+1,group+1))
        for i in range(group+1):
            if i == 0:
                H[i,0]=(rho_init-beta_sum)/Lambda
                H[i,1:(group+1)]=lamb
            else:
                H[i,0]=beta[i-1]/Lambda
                H[i,i]=-lamb[i-1]    
        
        self.H_matrix = H

        n = np.zeros(group+1)
        n[0] = self.neutron_density_init
        n[1:group+1] = self.precursor_conc_init
        self.n_vector_init = n

    def __update_H_matrix(self,t):
        self.H_matrix[0,0] = (self.rho_func(t)-self.total_beta)/self.neutron_life

    def ODE_func(self,t,y):
        self.H_matrix[0,0] = (self.rho_func(t)-self.total_beta)/self.neutron_life
        return np.dot(self.H_matrix, y)

    # Normal API
    # Raudu
    def normal_solve(self,step_control=True,max_step=1e-2):
        if step_control:
            max_step_control = (self.time_end-self.time_start)/10
            max_step = min(max_step,max_step_control)

        return_stuff = {}

        self.__define_H_matrix()
        results = solve_ivp(fun=self.ODE_func,t_span=(self.time_start,self.time_end),y0=self.n_vector_init,method='Radau',max_step=max_step)

        return_stuff['fine steps'] = results.t
        return_stuff['neutron density'] = results.y[0,:]
        return_stuff['precursor density'] = results.y[1:(self.delay_group_num+2),:]

        return return_stuff

    # Implicit Euler
    def normal_Euler_implicit_solve(self,step_control,time_step=1e-2):
        # Implicit Euler method
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize vector
        y = np.zeros((self.delay_group_num+1,num_steps+1))

        y[0,0] = self.neutron_density_init
        y[1:(self.delay_group_num+1),0] = self.precursor_conc_init

        # Initialize matrix
        self.__define_H_matrix()
        eye = np.identity(self.delay_group_num+1)
        # omega_series[0] = self.init_omega
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]

            # Update H matrix
            self.__update_H_matrix(t)

            # Solving
            y[:,i+1] = solve(eye-delta_t*self.H_matrix,y[:,i])
        n_density = y[0,:]
        C_density = y[1:(self.delay_group_num+1),:]

        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density

        return return_stuff
    # Fake SCM
    def normal_SCM_solve(self,step_control,time_step=1e-2):
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize
        n_density = np.zeros(num_steps+1)
        C_density = np.zeros((self.delay_group_num,num_steps+1))
        omega_series = np.zeros(num_steps+1)
        mu_series = np.zeros((self.delay_group_num,num_steps+1))

        n_density[0] = self.neutron_density_init
        C_density[:,0] = self.precursor_conc_init
        omega_series[0] = self.init_omega

        # SCM PKE solving
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]
            rho = self.rho_func(t)
            rho_last = self.rho_func(time_series[i])
            average_rho = (rho_last+rho)/2
            # Initilize neutron and precursor density
            n = n_density[i]
            C = C_density[:,i].copy()
            omega = omega_series[i]
            average_omega = omega
            for j in range(50):
                # Precursor update
                for group in range(self.delay_group_num):
                    C[group] = C_density[group,i]*np.exp(-self.delay_lambda[group]*delta_t)+\
                        (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t)*n_density[i]*\
                            (np.exp(average_omega*delta_t)-np.exp(-self.delay_lambda[group]*delta_t))/(average_omega+self.delay_lambda[group])
                
                # Neutron density update
                # n = n_density[i]*np.exp(average_omega*delta_t)
                precursor_term = (self.delay_lambda*(C+C_density[:,i])/2).sum()
                term_1 = np.exp(delta_t*(average_rho-self.total_beta)/self.neutron_life)*(n_density[i]+(self.neutron_life*precursor_term)/(average_rho-self.total_beta))
                term_2 = (self.neutron_life*precursor_term)/(average_rho-self.total_beta)
                n = term_1-term_2

                # Neutron density frequency update
                # precursor_term = self.delay_lambda*C
                # term_temp = 0
                for group in range(self.delay_group_num):
                    mu_series[group,i+1] = (self.delay_beta[group]*n)/(self.neutron_life*C[group])-self.delay_lambda[group]
                #     term_temp += self.delay_lambda[group]*self.delay_beta[group]/\
                #         (self.neutron_life*(mu_series[group,i+1]+self.delay_lambda[group]))
                # omega = (rho-self.total_beta)/self.neutron_life + precursor_term/n
                average_omega = np.log(n/n_density[i])/delta_t
                # omega = 2*average_omega-omega_series[i]
                omega = average_omega

                # print('Time {}, Iteration {}, omega: {}'.format(i,j,omega))
                if j>0 and abs((omega-omega_last)/omega)<=1e-4:
                    break
                omega_last = omega

            # Finish the current step     
            omega_series[i+1] = omega
            C_density[:,i+1] = C.copy()
            n_density[i+1] = n

        return_stuff['omega_series'] = omega_series
        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density
        return_stuff['precursor frequency'] = mu_series

        return return_stuff
    # SCM-order-1 
    def normal_SCM_secant_solve(self,step_control,time_step=1e-2):
        # Secant method
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize
        n_density = np.zeros(num_steps+1)
        C_density = np.zeros((self.delay_group_num,num_steps+1))
        omega_series = np.zeros(num_steps+1)
        mu_series = np.zeros((self.delay_group_num,num_steps+1))

        n_density[0] = self.neutron_density_init
        C_density[:,0] = self.precursor_conc_init
        omega_series[0] = self.init_omega

        # SCM PKE solving
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]
            rho = self.rho_func(t)
            # Initilize neutron and precursor density
            n0 = n_density[i]
            C0 = C_density[:,i].copy()
            C = C_density[:,i].copy()
            omega0 = omega_series[i]
            omega_guess = omega0
            for j in range(50):
                if j==0:
                    omega = omega_guess               
                # Update omega
                precursor_term = (self.delay_lambda*C).sum()
                n = n0*np.exp(delta_t*(omega0+omega)/2)
                # n = n0*np.exp(delta_t*omega)
                f = -omega + (rho-self.total_beta)/self.neutron_life+precursor_term/n
                if j==0:
                    if abs(omega_guess)>1e-2:
                        new_omega = omega_guess*1.01
                    else:
                        new_omega = omega_guess+0.01
                else:
                    new_omega = omega - (omega-omega_last)*f/(f-f_last)
        
                omega_last = omega
                omega = new_omega
                f_last = f
                if j>0 and abs(f)<=1e-3 and abs((omega-omega_last)/omega)<1e-4:
                    break
                average_omega = (omega0+omega)/2
                # average_omega = omega
                # Update precursor concentration
                # for group in range(self.delay_group_num):
                #     C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                #         (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t)*n0*\
                #             (np.exp(average_omega*delta_t)-np.exp(-self.delay_lambda[group]*delta_t))/(average_omega+self.delay_lambda[group])
                for group in range(self.delay_group_num):
                    C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                        (self.delay_beta[group]/(self.neutron_life*delta_t*self.delay_lambda[group]**2))*\
                            (delta_t*self.delay_lambda[group]*n+n0-n+\
                                np.exp(-self.delay_lambda[group]*delta_t)*(
                                    -delta_t*self.delay_lambda[group]*n0-n0+n
                                ))

            # Finish the current step     
            omega_series[i+1] = omega
            C_density[:,i+1] = C.copy()
            n_density[i+1] = n
            mu_series[:,i+1] = self.delay_beta*n/(self.neutron_life*C)-self.delay_lambda

        return_stuff['omega_series'] = omega_series
        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density
        return_stuff['precursor frequency'] = mu_series

        return return_stuff
    # SCM-order-1
    def normal_SCM_Newton_solve(self,step_control,time_step=1e-2):
        # Secant method
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize
        n_density = np.zeros(num_steps+1)
        C_density = np.zeros((self.delay_group_num,num_steps+1))
        omega_series = np.zeros(num_steps+1)
        mu_series = np.zeros((self.delay_group_num,num_steps+1))

        n_density[0] = self.neutron_density_init
        C_density[:,0] = self.precursor_conc_init
        omega_series[0] = self.init_omega

        # SCM PKE solving
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]
            rho = self.rho_func(t)
            # Initilize neutron and precursor density
            n0 = n_density[i]
            C0 = C_density[:,i].copy()
            C = C_density[:,i].copy()
            omega0 = omega_series[i]
            omega_guess = omega0
            for j in range(50):
                if j==0:
                    omega = omega_guess               
                # Update omega
                precursor_term = (self.delay_lambda*C).sum()
                n = n0*np.exp(delta_t*(omega0+omega)/2)
                f = -omega + (rho-self.total_beta)/self.neutron_life+precursor_term/n
                if j>0 and abs(f)<=1e-3:
                    break
                
                dif_f_1 = -1-precursor_term*delta_t/(2*n)
                new_omega = omega - f/dif_f_1
        
                omega = new_omega

                average_omega = (omega0+omega)/2
                # Update precursor concentration
                # for group in range(self.delay_group_num):
                #     C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                #         (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t)*n0*\
                #             (np.exp(average_omega*delta_t)-np.exp(-self.delay_lambda[group]*delta_t))/(average_omega+self.delay_lambda[group])
                for group in range(self.delay_group_num):
                    C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                        (self.delay_beta[group]/(self.neutron_life*delta_t*self.delay_lambda[group]**2))*\
                            (delta_t*self.delay_lambda[group]*n+n0-n+\
                                np.exp(-self.delay_lambda[group]*delta_t)*(
                                    -delta_t*self.delay_lambda[group]*n0-n0+n
                                ))


            # Finish the current step     
            omega_series[i+1] = omega
            C_density[:,i+1] = C.copy()
            n_density[i+1] = n
            mu_series[:,i+1] = self.delay_beta*n/(self.neutron_life*C)-self.delay_lambda

        return_stuff['omega_series'] = omega_series
        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density
        return_stuff['precursor frequency'] = mu_series

        return return_stuff

    # SCM-order-2
    def normal_SCM2_Newton_solve(self,step_control,time_step=1e-2):
        # 2nd-order SCM 
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize
        n_density = np.zeros(num_steps+1)
        C_density = np.zeros((self.delay_group_num,num_steps+1))
        omega_series = np.zeros(num_steps+1)
        mu_series = np.zeros((self.delay_group_num,num_steps+1))

        # For test
        omega_series_test = np.zeros(2*num_steps+1)
        time_series_test = np.linspace(self.time_start, self.time_end, num=2*num_steps+1)
        omega_series_test[0] = self.init_omega


        n_density[0] = self.neutron_density_init
        C_density[:,0] = self.precursor_conc_init
        omega_series[0] = self.init_omega

        # Jacobian matrix
        J_mat = np.zeros((2,2))
        F_vec = np.zeros(2)
        omega_vec = np.zeros(2)
        self.quad_coef = np.zeros(3)

        # SCM PKE solving
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]
            rho = self.rho_func(t)
            rho_1 = self.rho_func(t-delta_t/2)
            # Initilize neutron and precursor density
            n0 = n_density[i]
            C0 = C_density[:,i].copy()
            C = C_density[:,i].copy()
            C_1 = C_density[:,i].copy()
            omega0 = omega_series[i]
            omega_guess = omega0

            for j in range(50):
                if j==0:
                    omega = omega_guess   
                    omega_1 = omega_guess
                    omega_vec[0] = omega_1
                    omega_vec[1] = omega            
                # Update omega
                # t=h/2 (temp)
                precursor_term_1 = (self.delay_lambda*C_1).sum()
                n_1 = n0*np.exp(delta_t*(5*omega0+8*omega_1-omega)/24)
                # print('{},{},n0 {}'.format(i,j,n0))
                # print('{},{},n1 {}'.format(i,j,n_1))
                F_vec[0] = -omega_1 + (rho_1-self.total_beta)/self.neutron_life+precursor_term_1/n_1
                # t=h
                precursor_term = (self.delay_lambda*C).sum()
                n = n0*np.exp(delta_t*(omega0+4*omega_1+omega)/6)
                # print('{},{},n {}'.format(i,j,n))
                F_vec[1] = -omega + (rho-self.total_beta)/self.neutron_life+precursor_term/n
                if j>0 and np.abs(F_vec).max()<=1e-3:
                    break
                
                J_mat[0,0] = -1-precursor_term_1*delta_t/(3*n_1)
                J_mat[0,1] = precursor_term_1*delta_t/(24*n_1)
                J_mat[1,0] = -2*precursor_term*delta_t/(3*n)
                J_mat[1,1] = -1-precursor_term*delta_t/(6*n)
                
                omega_vec = omega_vec-np.dot(np.linalg.inv(J_mat),F_vec)

                omega_1 = omega_vec[0]
                omega = omega_vec[1]

                # self.quad_coef[0] = (2*n0-4*n_1+2*n)/(delta_t**2)
                # self.quad_coef[1] = (-3*n0+4*n_1-n)/delta_t
                # self.quad_coef[2] = n0
                self.quad_coef[0] = (2*omega0-4*omega_1+2*omega)/(delta_t**2)
                self.quad_coef[1] = (-3*omega0+4*omega_1-omega)/delta_t
                self.quad_coef[2] = omega0

                # Update precursor concentration
                # for group in range(self.delay_group_num):
                #     C_1[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t/2)+\
                #         (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t/2)*n0*\
                #             (np.exp(average_omega_1*delta_t/2)-np.exp(-self.delay_lambda[group]*delta_t/2))/(average_omega_1+self.delay_lambda[group])
                #     C[group] = C_1[group]*np.exp(-self.delay_lambda[group]*delta_t/2)+\
                #         (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t/2)*n0*\
                #             (np.exp(average_omega_2*delta_t/2)-np.exp(-self.delay_lambda[group]*delta_t/2))/(average_omega_2+self.delay_lambda[group])

                for group in range(self.delay_group_num):
                    self.compute_group = group
                    integral_1,err = integrate.quad(self.omega_func,0,delta_t/2)
                    integral_2,err = integrate.quad(self.omega_func,delta_t/2,delta_t)
                    integral_total = integral_1+integral_2
                    C_1[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t/2)+\
                        (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t/2)*\
                            n0*integral_1
                    C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                        (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t)*\
                            n0*integral_total

            # Finish the current step     
            omega_series[i+1] = omega
            C_density[:,i+1] = C.copy()
            n_density[i+1] = n
            mu_series[:,i+1] = self.delay_beta*n/(self.neutron_life*C)-self.delay_lambda

            omega_series_test[2*(i+1)] = omega
            omega_series_test[2*(i+1)-1] = omega_1

        return_stuff['omega_series'] = omega_series
        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density
        return_stuff['precursor frequency'] = mu_series

        # For test
        return_stuff['omega_series_test'] = omega_series_test
        return_stuff['fine steps test'] = time_series_test

        return return_stuff

    # SCM-order-0
    def normal_SCM0_Newton_solve(self,step_control,time_step=1e-2):
        # Secant method
        if step_control:
            time_step = min((self.time_end-self.time_start),time_step)
        else:
            time_step = (self.time_end-self.time_start)/10
        
        return_stuff = {}

        num_steps = int(np.round((self.time_end-self.time_start)/time_step))
        time_series = np.linspace(self.time_start, self.time_end, num=num_steps+1)

        # Initialize
        n_density = np.zeros(num_steps+1)
        C_density = np.zeros((self.delay_group_num,num_steps+1))
        omega_series = np.zeros(num_steps+1)
        mu_series = np.zeros((self.delay_group_num,num_steps+1))

        n_density[0] = self.neutron_density_init
        C_density[:,0] = self.precursor_conc_init
        omega_series[0] = self.init_omega

        # SCM PKE solving
        for i in range(num_steps):
            t = time_series[i+1]
            delta_t = time_series[i+1]-time_series[i]
            rho = self.rho_func(t)
            # Initilize neutron and precursor density
            n0 = n_density[i]
            C0 = C_density[:,i].copy()
            C = C_density[:,i].copy()
            # omega0 = omega_series[i]
            omega_guess = omega_series[i]
            for j in range(50):
                if j==0:
                    omega = omega_guess               
                # Update omega
                precursor_term = (self.delay_lambda*C).sum()
                n = n0*np.exp(delta_t*omega)
                f = -omega + (rho-self.total_beta)/self.neutron_life+precursor_term/n
                if j>0 and abs(f)<=1e-3:
                    break
                
                dif_f_1 = -1-precursor_term*delta_t/n
                new_omega = omega - f/dif_f_1
        
                omega = new_omega

                # average_omega = (omega0+omega)/2
                # Update precursor concentration
                # for group in range(self.delay_group_num):
                #     C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                #         (self.delay_beta[group]/self.neutron_life)*np.exp(-self.delay_lambda[group]*delta_t)*n0*\
                #             (np.exp(average_omega*delta_t)-np.exp(-self.delay_lambda[group]*delta_t))/(average_omega+self.delay_lambda[group])
                for group in range(self.delay_group_num):
                    C[group] = C0[group]*np.exp(-self.delay_lambda[group]*delta_t)+\
                        (self.delay_beta[group]/(self.neutron_life*delta_t*self.delay_lambda[group]**2))*\
                            (delta_t*self.delay_lambda[group]*n+n0-n+\
                                np.exp(-self.delay_lambda[group]*delta_t)*(
                                    -delta_t*self.delay_lambda[group]*n0-n0+n
                                ))


            # Finish the current step     
            omega_series[i+1] = omega
            C_density[:,i+1] = C.copy()
            n_density[i+1] = n
            mu_series[:,i+1] = self.delay_beta*n/(self.neutron_life*C)-self.delay_lambda

        return_stuff['omega_series'] = omega_series
        return_stuff['fine steps'] = time_series
        return_stuff['neutron density'] = n_density
        return_stuff['precursor density'] = C_density
        return_stuff['precursor frequency'] = mu_series

        return return_stuff

    # PKE solver API for time-spatial solver

    def predictor_SCM_solve(self,step_control,time_step=1e-2,solver='Normal'):
        
        if solver=='Normal':
            results = self.normal_SCM_solve(step_control=step_control,time_step=time_step)
        elif solver=='Secant':
            results = self.normal_SCM_secant_solve(step_control=step_control,time_step=time_step)
        elif solver=='Newton':
            results = self.normal_SCM_Newton_solve(step_control=step_control,time_step=time_step)
        return_stuff = {}

        parts_num = len(results['fine steps'])
        eval_time_step = time_step

        N_series = results['neutron density']
        omega_series = np.zeros(parts_num-2)
        omega_dif_2_series = np.zeros(parts_num-4)
        # Evaluate amplitude frequency
        for i in range(parts_num-2):
            omega_series[i] = np.log(N_series[i+2]/N_series[i])/(2*eval_time_step)
        for i in range(parts_num-4):
            omega_dif_2_series[i] = (omega_series[i]-2*omega_series[i+1]+omega_series[i+2])/(eval_time_step**2)

        # Find the element with the maximum absolute value
        abs_omega_dif_2_series = np.abs(omega_dif_2_series)
        i_max = np.where(abs_omega_dif_2_series==abs_omega_dif_2_series.max())
        omega_dif_2 = omega_dif_2_series[i_max[0][0]]

        omega = np.log(N_series[-1]/N_series[0])/(self.time_end-self.time_start)

        return_stuff['omega'] = omega
        return_stuff['omega_dif_2'] = omega_dif_2
        # return_stuff['fine steps'] = results['fine steps']
        return_stuff['neutron density'] = N_series[-1]
        return_stuff['precursor conc'] = results['precursor density'][:,-1]

        return return_stuff

    def predictor_solve(self,step_control=True,max_time_step=1e-2,time_eval=False):
        # Predict the amplitude frequency using PKE
        # The predicted amplitude frequency will be used as the initial value in the next iteration
        if step_control:
            max_step_control = (self.time_end-self.time_start)/10
            max_step = min(max_time_step,max_step_control)
        else:
            max_step = max_time_step
            
        self.__define_H_matrix()

        return_stuff = {}

        if not time_eval:
            results = solve_ivp(fun=self.ODE_func,t_span=(self.time_start,self.time_end),y0=self.n_vector_init,method='Radau',max_step=max_step)
            # Neutron density and precursor concentration at the end of the time step
            n_new = results.y[0,-1]
            omega = np.log(n_new/self.neutron_density_init)/(self.time_end-self.time_start)
            # C_new = results.y[1:(self.delay_group_num+1),-1]
            # # Precursor frequency
            # mu = self.delay_beta*n_new/(self.neutron_life*C_new)-self.delay_lambda
            # # Amplitude frequency
            # right_term = self.delay_lambda*self.delay_beta/(self.neutron_life*(mu+self.delay_lambda))
            # omega = (self.end_rho-self.total_beta)/self.neutron_life+right_term.sum()
            return_stuff['omega'] = omega
        else:
            parts_num = 11
            time_series = np.linspace(self.time_start, self.time_end, num=parts_num)
            eval_time_step = (self.time_end-self.time_start)/(parts_num-1)
            results = solve_ivp(fun=self.ODE_func,t_span=(self.time_start,self.time_end),y0=self.n_vector_init,method='Radau',t_eval=time_series,max_step=max_step)

            N_series = results.y[0,:]
            omega_series = np.zeros(parts_num-2)
            omega_dif_2_series = np.zeros(parts_num-4)
            # Evaluate amplitude frequency
            for i in range(parts_num-2):
                omega_series[i] = np.log(N_series[i+2]/N_series[i])/(2*eval_time_step)
            for i in range(parts_num-4):
                omega_dif_2_series[i] = (omega_series[i]-2*omega_series[i+1]+omega_series[i+2])/(eval_time_step**2)

            # Find the element with the maximum absolute value
            abs_omega_dif_2_series = np.abs(omega_dif_2_series)
            i_max = np.where(abs_omega_dif_2_series==abs_omega_dif_2_series.max())
            omega_dif_2 = omega_dif_2_series[i_max[0][0]]

            omega = np.log(N_series[-1]/N_series[0])/(self.time_end-self.time_start)

            return_stuff['omega'] = omega
            return_stuff['omega_dif_2'] = omega_dif_2
            # return_stuff['fine steps'] = time_series
            return_stuff['neutron density'] = N_series[-1]
            return_stuff['precursor conc'] = results.y[1:(self.delay_group_num+1),-1]

        return return_stuff


    def quad_exp_int(self,quad_coef,exp_coef,T):

        A = quad_coef[0]
        B = quad_coef[1]
        C = quad_coef[2]

        l = exp_coef

        term_1 = -(2-np.exp(T*l)*((T*l)**2-2*T*l+2))/l**3
        term_2 = (1+np.exp(T*l)*(T*l-1))/l**2
        term_3 = (np.exp(T*l)-1)/l

        return A*term_1+B*term_2+C*term_3

    def omega_func(self,t):
        return np.exp(self.quad_coef[0]/3*t**3+self.quad_coef[1]/2*t**2+\
            self.quad_coef[2]*t+self.delay_lambda[self.compute_group]*t)
