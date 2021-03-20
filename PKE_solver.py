import numpy as np
import scipy
from scipy.integrate import solve_ivp
'''
Point kinetics equations solver
Wei Xiao
Shanghai Jiao Tong University
School of Nuclear Science and Engineering
bearsanxw@gmail.com
2021-3-15
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

    def ODE_func(self,t,y):
        self.H_matrix[0,0] = (self.rho_func(t)-self.total_beta)/self.neutron_life
        return np.dot(self.H_matrix, y)

    # Solver API
    def normal_solve(self,step_control=True,max_step=1e-2):
        if step_control:
            max_step_control = (self.time_end-self.time_start)/10
            max_step = min(max_step,max_step_control)

        self.__define_H_matrix()
        results = solve_ivp(fun=self.ODE_func,t_span=(self.time_start,self.time_end),y0=self.n_vector_init,method='Radau',max_step=max_step)
        return results

    # PKE solver API for the convergence acceleration of SCM
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
            omega_dif_2 = np.max(omega_dif_2_series)
            omega = np.log(N_series[-1]/N_series[0])/(self.time_end-self.time_start)

            return_stuff['omega'] = omega
            return_stuff['omega_dif_2'] = omega_dif_2

        return return_stuff

    # PKE solver based on SCM
    def predictor_SCM_solve(self,step_control=True,max_time_step=1e-2):
        if step_control:
            max_step_control = (self.time_end-self.time_start)/10
            max_step = min(max_time_step,max_step_control)
        else:
            max_step = max_time_step
        
        return_stuff = {}

    def __SCM_solver(self):
        pass
