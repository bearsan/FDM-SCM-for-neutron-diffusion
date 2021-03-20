import numpy as np
import scipy
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
from collections import Iterable
import os
import copy

# Import PKE solvers
from PKE_solver import solver_PKE

'''
Transient Neutron Diffusion Equations Solver
Author: Wei Xiao
Shanghai Jiao Tong University
School of Nuclear Science and Engineering
bearsanxw@gmail.com
2021-3-18
'''
'''
This is the test code for transient algorithms.
The code is based on finite difference method (FDM) and 
stiffness confinement method (SCM).
'''
#TODO
# Use dataframe from pandas to storage the material
class precursor_FDM():
    def __init__(self,delay_group_num):
        self.delay_group_num = delay_group_num
        self.beta = np.zeros(delay_group_num)
        self.Lambda = np.zeros(delay_group_num)
    def set_beta(self,beta,group=None):
        if isinstance(beta, Iterable):
            if len(beta)!=self.delay_group_num:
                print('Group number error')
            else:
                self.beta = beta
        else:
            if group==None:
                print('Please input the group index')
            else:
                self.beta[group] = beta
    def set_lambda(self,Lambda,group=None):
        if isinstance(Lambda, Iterable):
            if len(Lambda)!=self.delay_group_num:
                print('Group number error')
            else:
                self.Lambda = Lambda
        else:
            if group==None:
                print('Please input the group index')
            else:
                self.Lambda[group] = Lambda

    def check_precursor(self):
        print('Delayed neutron group number: {}'.format(self.delay_group_num))
        print('Delayed neutron share:')
        print(self.beta)
        print('Decay constant:')
        print(self.Lambda)

class material_FDM:
    def __init__(self,mat_id):
        self.__id = mat_id
        self.__XS_name = ['XS_scatter','XS_fission_spectrum','XS_absorption','Diffusion_coef','XS_nu_fission']
        self.__XS = {}
        for name in self.__XS_name:
            self.__XS[name] = None
    def check_id(self):
        return self.__id
    def check_XS(self):
        print('Material {}'.format(self.__id))
        for name in self.__XS_name:
            print('{}:'.format(name))
            print(self.__XS[name])
    def set_XS(self,name,XS):
        if name not in self.__XS_name:
            print('Name error. There is no such name as {}'.format(name))
        else:
            self.__XS[name] = XS
    def get_XS(self,name):
        return self.__XS[name]

class geometry_FDM:
    def __init__(self,x_block,y_block):
        self.x_block = x_block
        self.y_block = y_block
        self.__mat_block = np.zeros((x_block,y_block),dtype=np.int32)
        self.x_dim = None
        self.y_dim = None
        self.x_size = None
        self.y_size = None
    
    def set_block_mat(self,mat,row,col):
        self.__mat_block[row,col] = mat.check_id()
    
    def set_block_mat_by_array(self,array):
        if array.shape[0] == self.x_block and array.shape[1] == self.y_block:
            self.__mat_block = array
        else:
            print('Dimension error')
    
    def set_block_size(self,x_size,y_size):
        if len(x_size)==self.x_block and len(y_size)==self.y_block:
            self.x_size = x_size
            self.y_size = y_size
        else:
            print('The number of block size is inconsistent with blocks layout')
    
    def set_discretized_num(self,x_dim,y_dim):
        if len(x_dim)==self.x_block and len(y_dim)==self.y_block:
            self.x_dim = x_dim
            self.y_dim = y_dim
        else:
            print('The number of discretized blocks is inconsistent with blocks layout')

    def check_blocks(self):
        print('Material layout:')
        print(self.__mat_block)
        print('Block size:')
        print('x:{}'.format(self.x_size))
        print('y:{}'.format(self.y_size))
        print('Discretization number:')
        print('x:{}'.format(self.x_dim))
        print('y:{}'.format(self.y_dim))

    def get_block_mat(self,row,col):
        return self.__mat_block[row,col]

class solver_FDM:
    def __init__(self,folder_name,group_num):
        self.__group_num = group_num
        self.__folder_name = folder_name
        self.__XS_ls = {}

        self.__boundary = {}
        self.__boundary_beta = {}

        self.boundary_set('left','reflective',1)
        self.boundary_set('bottom','reflective',1)
        self.boundary_set('right','vaccuum')
        self.boundary_set('top','vaccuum')

        self.__source_b = [None]*self.__group_num
        self.__fission_source = [None]*self.__group_num
        self.__matrix_A = [None]*self.__group_num

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    def add_material(self,material):
        if isinstance(material, Iterable):
            for mat in material:
                if mat.check_id() not in self.__XS_ls:
                    XS = self.__to_XS(mat)
                    self.__XS_ls[mat.check_id()] = XS
                else:
                    print('Material {} has existed'.format(mat.check_id()))
        else:
            if material.check_id() not in self.__XS_ls:
                XS = self.__to_XS(material)
                self.__XS_ls[material.check_id()] = XS
            else:
                print('Material {} has existed'.format(material.check_id()))
    
    #TODO
    def update_material(self,mat_id,material,mat_type=None):
        if mat_type == None:
            if mat_id in self.__XS_ls:
                XS = self.__to_XS(material)
                self.__XS_ls[mat_id] = XS
            else:
                print('There is no material {}'.format(mat_id))
        elif mat_type == 'real':
            XS = self.__to_XS(material)
            self.__XS_ls_real[mat_id] = XS
    
    # Material list
    # 1st: material; 2nd: group; 3rd: XS: Dg, Σrg, vΣfg, Σs, χ    
    def __to_XS(self,material):
        XS = [None]*self.__group_num
        for group in range(self.__group_num):
            XS_group = [None]*5
            XS_group[0] = material.get_XS('Diffusion_coef')[group]
            XS_group[2] = material.get_XS('XS_nu_fission')[group]
            XS_group[3] = material.get_XS('XS_scatter')[group]
            XS_group[4] = material.get_XS('XS_fission_spectrum')[group]
            # Get the removal cross section
            for i in range(self.__group_num):
                if i==0:
                    XS_group[1] = material.get_XS('XS_absorption')[group]
                else:
                    if i != group:
                        XS_group[1] += XS_group[3][i]
            XS[group] = XS_group
        return XS

    # Set precursor
    def set_precursor(self,precursor):

        self.__delay_group_num = precursor.delay_group_num
        self.__delay_beta = precursor.beta
        self.__delay_lambda = precursor.Lambda
        self.__delay_total_beta = 0
        for i in range(self.__delay_group_num):
            self.__delay_total_beta += precursor.beta[i]

    def set_neutron_velocity(self,vel):
        if len(vel)==self.__group_num:
            self.__neutron_v = vel
        else:
            print('Group number error')


    # Geometry
    def set_geometry(self,geo,transient_mode=False):
        
        self.__block_row = geo.x_block
        self.__block_col = geo.y_block 
        # Initial row and column number
        # row (x)
        self.__mat_row = 0
        self.__x_mat_index = [0]*(geo.x_block+1)
        for i,x_dim in enumerate(geo.x_dim):
            self.__mat_row += x_dim
            self.__x_mat_index[i+1] = x_dim+self.__x_mat_index[i]
        # Vacuum boundary
        self.__x_mat_index[-1] -= 1
        self.__mat_row -= 1
        
        # column (y)
        self.__mat_col = 0
        self.__y_mat_index = [0]*(geo.y_block+1)
        for i,y_dim in enumerate(geo.y_dim):
            self.__mat_col += y_dim
            self.__y_mat_index[i+1] = y_dim+self.__y_mat_index[i]
        # Vacuum boundary
        self.__y_mat_index[-1] -= 1
        self.__mat_col -= 1

        self.__flux_row = self.__mat_row+1
        self.__flux_col = self.__mat_col+1    

        # Set pseudo material
        if transient_mode:
            self.__define_pseudo_material(geo)
        # Designate material id to blocks
        self.__set_mat_block(geo,transient_mode)

        # Initial material layout
        self.__define_mat_m()
        self.__define_geo_m_x(geo)
        self.__define_geo_m_y(geo)
        self.calculate_mesh_area()
        self.calculate_block_area()
        np.savetxt('{}/material_layout.asu'.format(self.__folder_name),self.mat_m,fmt="%i")

    # def update_geo_mat(self,geo,material):
    #     # Update material and geometry for transient cases
    #     self.__XS_ls = {}
    #     for mat in material:
    #         XS = self.__to_XS(mat)
    #         self.__XS_ls[mat.check_id()] = XS
    #     self.__define_pseudo_material(geo)
    #     self.__set_mat_block(geo,True)
    #     # Initial material layout
    #     self.__define_mat_m()
    #     self.__define_geo_m_x(geo)
    #     self.__define_geo_m_y(geo)
    #     self.calculate_mesh_area()
    #     self.calculate_block_area()
    #     np.savetxt('{}/material_layout.asu'.format(self.__folder_name),self.mat_m,fmt="%i")

    # Index transfer
    def __v2m(self,v_i):
        m_j = v_i//self.__flux_row
        m_i = v_i-m_j*self.__flux_row
        return m_i,m_j
    def __m2v(self,m_i,m_j):
        v_i = m_j*self.__flux_row+m_i
        return v_i
    def __v2m_block(self,v_i):
        m_j = v_i//self.__block_row
        m_i = v_i-m_j*self.__block_col
        return m_i,m_j
    def __m2v_block(self,m_i,m_j):
        v_i = m_j*self.__block_row+m_i
        return v_i

    def __set_mat_block(self,geo,transient_mode):
        # If it is transient mode, every block has the unique material
        # The real material and dynamics material should be recorded both
        self.__mat_block = np.zeros((self.__block_row,self.__block_col),dtype=np.int32)
        self.__mat_block_real = np.zeros((self.__block_row,self.__block_col),dtype=np.int32)
        if not transient_mode:
            for j in range(self.__block_col):
                for i in range(self.__block_row):
                    self.__mat_block[i,j] =  geo.get_block_mat(i,j)
        else:
            k = 0
            for j in range(self.__block_col):
                for i in range(self.__block_row):
                    self.__mat_block[i,j] =  k
                    self.__mat_block_real[i,j] = geo.get_block_mat(i,j)
                    k += 1
    
    def __define_pseudo_material(self,geo):
        block_number = self.__block_row*self.__block_col
        self.__XS_ls_real = copy.deepcopy(self.__XS_ls)
        self.__XS_ls = [None]*block_number
        for j in range(self.__block_col):
            for i in range(self.__block_row):
                mat_id_real = geo.get_block_mat(i,j)
                k = j*self.__block_row+i
                self.__XS_ls[k] = copy.deepcopy(self.__XS_ls_real[mat_id_real])

    # For test
    def get_XS(self):
        return self.__XS_ls_real

    def __define_mat_m(self):
        self.mat_m = np.zeros((self.__mat_row,self.__mat_col),dtype=np.int32)
        for j in range(self.__block_col):
            for i in range(self.__block_row):
                self.mat_m[self.__x_mat_index[i]:self.__x_mat_index[i+1],
                self.__y_mat_index[j]:self.__y_mat_index[j+1]] = self.__mat_block[i,j]

    #TODO
    def __define_geo_m_x(self,geo):
        self.geo_m_x = np.zeros(self.__mat_row)
        for i in range(geo.x_block):
            self.geo_m_x[self.__x_mat_index[i]:self.__x_mat_index[i+1]] = geo.x_size[i]/geo.x_dim[i]
    
    def __define_geo_m_y(self,geo):
        self.geo_m_y = np.zeros(self.__mat_col)
        for i in range(geo.y_block):
            self.geo_m_y[self.__y_mat_index[i]:self.__y_mat_index[i+1]] = geo.y_size[i]/geo.y_dim[i]

    def calculate_mesh_area(self):
        # point wise area
        self.__x_mesh_area = np.zeros(self.__flux_row)
        self.__x_mesh_area[0] = self.geo_m_x[0]*0.5
        self.__x_mesh_area[-1] = self.geo_m_x[-1]*0.5
        self.__x_mesh_area[1:(self.__flux_row-1)] = 0.5*(self.geo_m_x[1:(self.__mat_row)]+\
            self.geo_m_x[0:(self.__mat_row-1)])

        self.__y_mesh_area = np.zeros(self.__flux_col)
        self.__y_mesh_area[0] = self.geo_m_y[0]*0.5
        self.__y_mesh_area[-1] = self.geo_m_y[-1]*0.5
        self.__y_mesh_area[1:(self.__flux_col-1)] = 0.5*(self.geo_m_y[1:(self.__mat_col)]+\
            self.geo_m_y[0:(self.__mat_col-1)])
        
        temp = np.outer(self.__y_mesh_area,self.__x_mesh_area)
        # np.savetxt('area.asu',temp,fmt="%1.2f")
        self.__mesh_area = temp.flatten()

    def calculate_block_area(self):
        mesh_area = np.reshape(self.__mesh_area,(self.__flux_row,self.__flux_col),order='F')
        temp = np.zeros((self.__block_row,self.__block_col))
        for j in range(self.__block_col):
            for i in range(self.__block_row):
                temp[i,j] = mesh_area[self.__x_mat_index[i]:(self.__x_mat_index[i+1]+1),
                self.__y_mat_index[j]:(self.__y_mat_index[j+1]+1)].sum()
        self.__block_area = temp.flatten(order='F')

    # Initial flux
    def __initial_flux(self,flux_type='Normal'):
        if flux_type == 'Normal':
            #TODO
            # Initial flux storage
            self.__flux = [None]*self.__group_num
            self.__flux_blockwise = [None]*self.__group_num
            for i in range(self.__group_num):
                self.__flux[i] = np.ones(self.__flux_col*self.__flux_row)
                self.__flux_blockwise[i] = np.ones(self.__block_col*self.__block_row)
            self.__fission_dist = np.ones(self.__flux_col*self.__flux_row)
            self.__fission_dist_last = np.ones(self.__flux_col*self.__flux_row)
        elif flux_type == 'Adjoint':
            #TODO
            # Initial flux storage
            self.__flux_adjoint = [None]*self.__group_num
            self.__flux_blockwise_adjoint = [None]*self.__group_num
            for i in range(self.__group_num):
                self.__flux_adjoint[i] = np.ones(self.__flux_col*self.__flux_row)
                self.__flux_blockwise_adjoint[i] = np.ones(self.__block_col*self.__block_row)
            self.__fission_dist = np.ones(self.__flux_col*self.__flux_row)
            self.__fission_dist_last = np.ones(self.__flux_col*self.__flux_row)
    
    ############Initial settings for transient calculation############
    def initial_dynamics(self,time_steps,transient_algorithm='SCM',vtk_save=True,max_time_step=2e-2,min_time_step=1e-3):
        
        # Save results to .vtk or not
        self.__vtk_save = vtk_save
        # SCM
        if transient_algorithm=='SCM':
            # Time steps setting
            self.__time_steps = time_steps
            self.__num_time_steps = len(time_steps)
            # Initialize dynamics frequency and concentration 
            self.__initial_freq()
            # Get initial concentration
            for i in range(self.__delay_group_num):
                self.__precursor_conc[i] = self.__delay_beta[i]*self.__fission_dist_blockwise/\
                    (self.__precursor_freq[i]+self.__delay_lambda[i])
            self.__amp_freq_ls = np.zeros((2))
          
            # Initialize saving data 
            self.__results_reactivity = np.zeros(self.__num_time_steps)
            self.__results_Q = np.zeros(self.__num_time_steps)
            self.__results_amp_freq = np.zeros(self.__num_time_steps)
            self.__results_shape_freq = np.zeros((self.__group_num,self.__num_time_steps,self.__block_col*self.__block_row))
            self.__results_precursor_conc = np.zeros((self.__delay_group_num,self.__num_time_steps,self.__block_col*self.__block_row))
            self.__results_precursor_freq = np.zeros((self.__delay_group_num,self.__num_time_steps,self.__block_col*self.__block_row))
        
            self.__save_mat(time_index=0)
        # Adaptive SCM
        elif transient_algorithm=='SCM_adaptive':
            # Time steps setting
            self.__key_time_steps = time_steps
            # The maximum number of time steps (total time interval/minimum time step size)
            max_step_num = int((time_steps[-1]-time_steps[0])/min_time_step+1)
            self.__time_steps = np.zeros(max_step_num)
            self.__time_steps[0] = time_steps[0]
            self.__num_time_steps = max_step_num
            # The maximum and minimum time step size for  
            self.__max_time_step = max_time_step
            self.__min_time_step = min_time_step
            # The time pointer
            # Time pointer is used to make sure that there is no time interval containing the key time points
            self.__time_pointer = 1
            # Initialize dynamics frequency and concentration 
            self.__initial_freq()
            # Get initial concentration
            for i in range(self.__delay_group_num):
                self.__precursor_conc[i] = self.__delay_beta[i]*self.__fission_dist_blockwise/\
                    (self.__precursor_freq[i]+self.__delay_lambda[i])
            self.__amp_freq_ls = np.zeros((2))
          
            # Initialize saving data 
            self.__results_reactivity = np.zeros(self.__num_time_steps)
            self.__results_Q = np.zeros(self.__num_time_steps)
            self.__results_amp_freq = np.zeros(self.__num_time_steps)
            self.__results_shape_freq = np.zeros((self.__group_num,self.__num_time_steps,self.__block_col*self.__block_row))
            self.__results_precursor_conc = np.zeros((self.__delay_group_num,self.__num_time_steps,self.__block_col*self.__block_row))
            self.__results_precursor_freq = np.zeros((self.__delay_group_num,self.__num_time_steps,self.__block_col*self.__block_row))
        
            self.__save_mat(time_index=0,save_type='SCM_adaptive')
            pass
        # Implicit Euler
        elif transient_algorithm=='Implicit_Euler':
            # Time steps setting
            self.__time_steps = time_steps
            self.__num_time_steps = len(time_steps)

            self.__initial_freq(transient_algorithm=transient_algorithm)

            # Why????
            self.__fission_dist_pointwise = self.__fission_dist_pointwise/self.__k_init
            self.__fission_dist_blockwise = self.__fission_dist_blockwise/self.__k_init
            for group in range(self.__group_num):
                self.__flux[group] = self.__flux[group]/self.__k_init
                
            for i in range(self.__delay_group_num):
                self.__precursor_conc[i] = self.__delay_beta[i]*self.__fission_dist_blockwise/self.__delay_lambda[i]
        
        # Storage the initial XS
        self.__init_XS_narray = self.__narray_XS()


    # Initial dynamic frequency 
    def __initial_freq(self,transient_algorithm='SCM'):
        if transient_algorithm=='SCM':
            # Initial amplitude
            self.__amp = 1
            # Flux shape frequency
            self.__shape_freq = [None]*self.__group_num
            for i in range(self.__group_num):
                self.__shape_freq[i] = np.zeros(self.__block_row*self.__block_col)
            # Flux amplitude frequency
            # TODO
            # self.__amp_freq_2nd = 1
            self.__amp_freq = 0
            self.__amp_freq_last = 0
            # self.__initial_amp_freq()
            # Precursor frequency
            self.__precursor_freq = [None]*self.__delay_group_num
            for i in range(self.__delay_group_num):
                self.__precursor_freq[i] = np.zeros(self.__block_row*self.__block_col) 

        # Precursor concentration
        self.__precursor_conc = [None]*self.__delay_group_num
        for i in range(self.__delay_group_num):
            self.__precursor_conc[i] = np.zeros(self.__block_row*self.__block_col)  
    
    def __initial_amp_freq(self):
        self.__update_amp_freq(self.__k,self.__fission_dist_pointwise,self.__flux)

    def __narray_XS(self):
        # dict + narray
        # We don't store the diffusion coefficients
        narray_XS = {}
        nufission_block = np.zeros((self.__group_num,self.__block_row*self.__block_col))
        chi_block = np.zeros((self.__group_num,self.__block_row*self.__block_col))
        total_block = np.zeros((self.__group_num,self.__block_row*self.__block_col))
        scatter_block = np.zeros((self.__group_num,self.__group_num,self.__block_row*self.__block_col))
        for group in range(self.__group_num):
            for j in range(self.__block_col):
                for i in range(self.__block_row):
                    v_i = self.__m2v_block(i,j)
                    mat_id_real = self.__mat_block_real[i,j]
                    # XS
                    nufission_block[group,v_i] = self.__XS_ls_real[mat_id_real][group][2]/self.__k_init
                    chi_block[group,v_i] = self.__XS_ls_real[mat_id_real][group][4]
                    total_block[group,v_i] = self.__XS_ls_real[mat_id_real][group][1]
                    for group_in in range(self.__group_num):
                        scatter_block[group,group_in,v_i] = self.__XS_ls_real[mat_id_real][group][3][group_in]
        
        narray_XS['nu-fission'] = nufission_block
        narray_XS['chi'] = chi_block
        narray_XS['total'] = total_block
        narray_XS['scatter'] = scatter_block

        return narray_XS


    def __update_dynamics_XS(self,transient_algorithm='SCM',time_interval=None):
        if transient_algorithm=='SCM':
            # Update dynamics cross-sections
            # Temp variable (vector:block_row*block_col)
            self.__nufission_block = [None]*self.__group_num
            self.__chi_block = [None]*self.__group_num
            temp = np.zeros((self.__block_row*self.__block_col))
            for delay_group in range(self.__delay_group_num):
                temp += self.__delay_lambda[delay_group]*self.__delay_beta[delay_group]/\
                    (self.__precursor_freq[delay_group]+self.__delay_lambda[delay_group])
            for group in range(self.__group_num):
                self.__nufission_block[group] = np.zeros(self.__block_row*self.__block_col)
                self.__chi_block[group] = np.zeros(self.__block_row*self.__block_col)
                for j in range(self.__block_col):
                    for i in range(self.__block_row):
                        v_i = self.__m2v_block(i,j)
                        mat_id = self.__mat_block[i,j]
                        mat_id_real = self.__mat_block_real[i,j]
                        # Dynamics fission XS
                        temp_var = self.__XS_ls_real[mat_id_real][group][2]/self.__k_init
                        self.__XS_ls[mat_id][group][2] = temp_var
                        self.__nufission_block[group][v_i] =  temp_var
                        # Dynamics absorption XS
                        self.__XS_ls[mat_id][group][1] = self.__XS_ls_real[mat_id_real][group][1]+\
                            (self.__shape_freq[group][v_i]+self.__amp_freq)/self.__neutron_v[group]
                        # Dynamics fission spectrum
                        self.__XS_ls[mat_id][group][4] = self.__XS_ls_real[mat_id_real][group][4]*\
                            (1-self.__delay_total_beta+temp[v_i])
                        self.__chi_block[group][v_i] = self.__XS_ls_real[mat_id_real][group][4]
        elif transient_algorithm=='Implicit_Euler':
            temp = 0
            # print(self.__XS_ls_real)
            self.__chi_block = [None]*self.__group_num
            for delay_group in range(self.__delay_group_num):
                temp += self.__delay_lambda[delay_group]*self.__delay_beta[delay_group]*time_interval/\
                    (1+self.__delay_lambda[delay_group]*time_interval)
            for group in range(self.__group_num):
                chi_block_group = np.zeros((self.__block_row*self.__block_col))
                for j in range(self.__block_col):
                    for i in range(self.__block_row):
                        v_i = self.__m2v_block(i,j)
                        mat_id = self.__mat_block[i,j]
                        mat_id_real = self.__mat_block_real[i,j]
                        # Dynamics fission XS
                        self.__XS_ls[mat_id][group][2] = self.__XS_ls_real[mat_id_real][group][2]/self.__k_init
                        # Dynamics absorption XS
                        self.__XS_ls[mat_id][group][1] = self.__XS_ls_real[mat_id_real][group][1]+1/(self.__neutron_v[group]*time_interval)
                        # Dynamics fission spectrum
                        self.__XS_ls[mat_id][group][4] = self.__XS_ls_real[mat_id_real][group][4]*(1-self.__delay_total_beta+temp)
                        # print(chi_block_group[v_i])
                        chi_block_group[v_i] = self.__XS_ls_real[mat_id_real][group][4]
                self.__chi_block[group] = chi_block_group

        # print('Pseudo XS')
        # print(self.__XS_ls)
        # print('Real XS')
        # print(self.__XS_ls_real)

    def __update_conc(self,fission_blockwise,fission_blockwise_last,precursor_conc_last,time_interval):
        # Update the concentration of precursors (with linear interpolation)
        # fission_dist_blockwise: independent of the energy group (real) 
        # fission_dist_blockwise_last: last time step (real)
        # precursor_conc_last: last time step
        # time_interval: Oh! 
        # Method 1
        # Integration
        for i in range(self.__delay_group_num):
        # fission integration over time interval
            Q_int = (fission_blockwise-fission_blockwise_last)*\
                ((time_interval-1/self.__delay_lambda[i])*np.exp(self.__delay_lambda[i]*time_interval)+1/self.__delay_lambda[i])\
                    /(time_interval*self.__delay_lambda[i])+\
                        fission_blockwise_last*(np.exp(self.__delay_lambda[i]*time_interval)-1)/self.__delay_lambda[i]
            self.__precursor_conc[i] = precursor_conc_last[i]*np.exp(-self.__delay_lambda[i]*time_interval)\
                +self.__delay_beta[i]*np.exp(-self.__delay_lambda[i]*time_interval)*Q_int
        # # Method 2
        # # Implicit Euler
        # for i in range(self.__delay_group_num):
        #     self.__precursor_conc[i] = (precursor_conc_last[i]+\
        #         fission_blockwise*time_interval*self.__delay_beta[i])/(1+self.__delay_lambda[i]*time_interval)
    def __update_conc_exp(self,flux_blockwise_last,precursor_conc_last,amp_freq,amp_freq_last,shape_freq,shape_freq_last,time_interval):

        # TODO
        ave_amp_freq = (amp_freq+amp_freq_last)/2
        ave_shape_freq = shape_freq

        Q_g = [None]*self.__group_num
        for group in range(self.__group_num):
            Q_g[group] = self.__nufission_block[group]*flux_blockwise_last[group]
        
        for i in range(self.__delay_group_num):
            integral = 0
            for group in range(self.__group_num):
                integral += Q_g[group]*(np.exp((ave_amp_freq+ave_shape_freq[group])*time_interval)-np.exp(-self.__delay_lambda[i]*time_interval))\
                    /(ave_amp_freq+ave_shape_freq[group]+self.__delay_lambda[i])
            self.__precursor_conc[i] = precursor_conc_last[i]*np.exp(-self.__delay_lambda[i]*time_interval)\
                +self.__delay_beta[i]*integral


    def __update_amp_freq(self,kD,fission_dist_pointwise,flux_pointwise):
        # Newton-Raphson method
        # TODO
        # kD: dynamics eigenvalue
        # fission_dist_pointwise: independent of the energy group
        # fission_source_pointwise: dependent of the energy group (fission spectrum)
        numerator = 0
        denominator = 0
        for group in range(self.__group_num):
            numerator += (fission_dist_pointwise*flux_pointwise[group]*\
                self.__mesh_area/self.__neutron_v[group]).sum()
        # denominator += (fission_dist_pointwise*fission_dist_pointwise*\
        #         self.__mesh_area).sum()
            denominator += (fission_dist_pointwise*self.__fission_source[group]).sum()
        # Update the amplitude frequency
        l = numerator/denominator
        print('Amplitude frequency: {}'.format(self.__amp_freq))
        self.__amp_freq += (1-1/kD)/l
        print('Amplitude frequency (new): {}'.format(self.__amp_freq))
        

    def __update_amp_freq_secant(self,kD,kD_last,amp_freq,amp_freq_last,iter_index):
        # Method 2
        # print('kD:{}'.format(kD))
        # print('kD last:{}'.format(kD_last))
        # if i==0:
        #     self.__amp_freq = self.__amp_freq_predict
        # elif i==1:
        #     self.__amp_freq_last = self.__amp_freq_predict
        #     if abs(self.__amp_freq_predict)>=1e-7:
        #         self.__amp_freq = 1.1*self.__amp_freq_predict
        #     else:
        #         self.__amp_freq = self.__amp_freq_predict + 0.1
        print('Amplitude frequency last:{}'.format(amp_freq_last))
        print('Amplitude frequency:{}'.format(amp_freq))
        if iter_index>=1:
            amp_freq_new = amp_freq+(amp_freq_last-amp_freq)*(1-kD)/(kD_last-kD)
            self.__amp_freq_last = amp_freq
            self.__amp_freq = amp_freq_new
        elif iter_index==0:
            if abs(amp_freq)<1e-7:
                amp_freq_new = amp_freq+0.1
            else:
                amp_freq_new = amp_freq*1.01
            self.__amp_freq_last = amp_freq
            self.__amp_freq = amp_freq_new
        print('Amplitude frequency new:{}'.format(amp_freq_new))

    
    def __update_shape_freq(self,flux_blockwise,flux_blockwise_last,shape_freq_last,time_interval):
        # flux should be normalized first
        # flux_blockwise: flux which is normalized to the amplitude from the last time step
        # flux_blockwise_last: real flux from the last time step
        # Method 1 
        # for group in range(self.__group_num):
        #     self.__shape_freq[group] = 2*np.log(flux_blockwise[group]/flux_blockwise_last[group])/time_interval-\
        #         shape_freq_last[group]
        # Method 2
        for group in range(self.__group_num):
            self.__shape_freq[group] = np.log(flux_blockwise[group]/flux_blockwise_last[group])/time_interval

    def __update_precursor_freq(self,precursor_conc,fission_blockwise):
        # precursor_conc: precursor concentration
        # fission_blockwise: fission distribution (real)
        for i in range(self.__delay_group_num):
            self.__precursor_freq[i] = self.__delay_beta[i]*fission_blockwise/precursor_conc[i]-self.__delay_lambda[i]
        

    def __update_flux_real(self,flux_pointwise,amp,amp_last,time_interval):
        # flux_pointwise: normalized flux from the current time step
        real_flux = [None]*self.__group_num
        # if time_index == 1:
        #     integral = time_interval*(amp+amp_last)/2
        # else:
        #     # For test
        #     y1 = np.array([amp])
        #     y = np.concatenate([self.__amp_freq_ls,y1])
        #     order = len(y)
        #     x = np.linspace(0.0, (order-1)*time_interval, num=order)
        #     p = np.polyfit(x,y,order-1)
        #     integral = 0
        #     t1 = (order-2)*time_interval
        #     t2 = (order-1)*time_interval
        #     for i in range(order):
        #         degree = order-1-i
        #         integral += (t2**(degree+1)-t1**(degree+1))*p[i]/(degree+1)
        integral = time_interval*(amp+amp_last)/2
        exp_integral = np.exp(integral) 
        for group in range(self.__group_num):
            real_flux[group] = exp_integral*flux_pointwise[group]
         
        return real_flux

    def __define_time_source(self,flux_blockwise,precursor_conc_last,time_interval,time_index):
        # print('Inertial source:')
        # Blockwise
        time_source = [None]*self.__group_num 
        # Pointwise
        self.__time_source = [None]*self.__group_num
        for group in range(self.__group_num):
            temp = np.zeros(self.__block_row*self.__block_col)
            for delay_group in range(self.__delay_group_num):
                temp += self.__chi_block[group]*self.__delay_lambda[delay_group]*precursor_conc_last[delay_group]/\
                        (1+self.__delay_lambda[delay_group]*time_interval)
            time_source[group] = flux_blockwise[group]/(self.__neutron_v[group]*time_interval)+temp
            # print('Group {}'.format(group))
            # print(time_source[group])
            # blockwise to pointwise
            self.__time_source[group] = self.__blockwise2pointwise(time_source[group])
            # if time_index==1:
            #     self.__time_source[group] = self.__blockwise2pointwise(time_source[group])/self.__k_init
            # else:
            #     self.__time_source[group] = self.__blockwise2pointwise(time_source[group])
                

    ########################################################################
    # PKE predictor for amplitude frequency
    def __get_reactivity(self,fission_blockwise,flux_blockwise,adjoint_flux_blockwise,XS,init_XS):
        # F
        F = 0
        for group in range(self.__group_num):
            integral = self.__block_area*adjoint_flux_blockwise[group]*fission_blockwise*XS['chi'][group,:]
            F += integral.sum()
        # delta absorption term
        total_term = 0
        for group in range(self.__group_num):
            temp = (XS['total'][group,:]-init_XS['total'][group,:])*self.__block_area*adjoint_flux_blockwise[group]*flux_blockwise[group]
            total_term += temp.sum()
        # delta scatter term
        scatter_term = 0
        for group_in in range(self.__group_num):
            temp = 0
            for group_out in range(self.__group_num):
                temp += (XS['scatter'][group_out,group_in,:]-init_XS['scatter'][group_out,group_in,:])*flux_blockwise[group_out]
            temp = temp*self.__block_area*adjoint_flux_blockwise[group_in]
            scatter_term += temp.sum()


        # delta fission term
        fission_term = 0
        for group_out in range(self.__group_num):
            temp = 0
            for group_in in range(self.__group_num):
                temp += XS['chi'][group_in,:]*adjoint_flux_blockwise[group_in]
            temp = temp*self.__block_area*flux_blockwise[group_out]*(XS['nu-fission'][group_out,:]-init_XS['nu-fission'][group_out,:])
            fission_term += temp.sum()

        rho = (-total_term + scatter_term + fission_term)/F
        return rho

    def __get_dynamics_parameters(self,time_index,fission_blockwise,flux_blockwise,adjoint_flux_blockwise,precursor_conc,XS,rho=0.0):
        # TODO
        dynamics_parameters = {}
        # F
        F = 0
        for group in range(self.__group_num):
            integral = self.__block_area*adjoint_flux_blockwise[group]*fission_blockwise*XS['chi'][group,:]
            F += integral.sum()
        
        if True:
            # Precursor concentration
            # Area weighted
            precursor_average = np.zeros(self.__delay_group_num)
            for i in range(self.__delay_group_num):
                temp = 0.0
                for group in range(self.__group_num):
                    temp += XS['chi'][group,:]*adjoint_flux_blockwise[group]*self.__precursor_conc[i]*self.__block_area
                precursor_average[i] = temp.sum()
            # # Neutron density
            neutron_density = 0.0
            for i in range(self.__group_num):
                temp = adjoint_flux_blockwise[i]*flux_blockwise[i]*self.__block_area/self.__neutron_v[i]
                neutron_density += temp.sum()
        else:
            neutron_density = self.__n_PKE[0]
            precursor_average = self.__n_PKE[1:(self.__group_num+1)]

        # Neutron generation time
        neutron_time = 0
        for group in range(self.__group_num):
            integral = self.__block_area*adjoint_flux_blockwise[group]*flux_blockwise[group]/self.__neutron_v[group]
            neutron_time += integral.sum()
        neutron_time = neutron_time/F
        # Delay fractions
        # Delay fractions are spatially independent here
        delay_fractions = np.zeros(self.__delay_group_num)
        for delay_group in range(self.__delay_group_num):
            delay_fractions[delay_group] = self.__delay_beta[delay_group]
        # Decay constants
        # TODO
        # Decay constants are spatially independent here
        decay_constants = np.zeros(self.__delay_group_num)
        for delay_group in range(self.__delay_group_num):
            decay_constants[delay_group] = self.__delay_lambda[delay_group]
        # # Normalized factor for precursor concentration
        # total_beta = delay_fractions.sum()
        # temp = decay_constants*precursor_average
        # k = (self.__amp_freq-(rho-total_beta)/neutron_time)/(temp.sum()/neutron_density)

        dynamics_parameters['neutron_time'] = neutron_time
        dynamics_parameters['delay_fractions'] = delay_fractions
        dynamics_parameters['decay_constants'] = decay_constants
        dynamics_parameters['neutron_density'] = neutron_density
        dynamics_parameters['precursor_conc'] = precursor_average

        return dynamics_parameters

    def __get_time_step(self,time_index,current_h,omega_dif_2,tolerance):
        # Error and optimized time step evaluation
        error = (current_h**3)*abs(omega_dif_2)/12
        print('Predicted error: {}'.format(error))
        optimized_h = np.cbrt(12*tolerance/abs(omega_dif_2))
        print('Optimized time step: {}'.format(optimized_h))

        final_step = max(self.__min_time_step,min(self.__max_time_step,optimized_h))

        if (final_step+self.__time_steps[time_index-1])>self.__key_time_steps[self.__time_pointer]:
            final_step = self.__key_time_steps[self.__time_pointer]-self.__time_steps[time_index-1]
            self.__time_pointer += 1
        self.__time_steps[time_index] = self.__time_steps[time_index-1]+final_step
        print('Choosen time step: {}'.format(final_step))

        return final_step

    ########################################################################

    # Boundary
    def boundary_set(self,loc,boundary_type,beta=0):
        if boundary_type=='reflective':
            self.__boundary[loc]=0
            self.__boundary_beta[loc] = beta
        else:
            self.__boundary[loc]=1

    def print_boundary(self):
        print('Boundary type')
        print(self.__boundary)

    # Discretization of operators
    #TODO
    def define_scattering_term(self,group_in,flux=None):
        if flux==None:
            flux = self.__flux
        # Validated 2021-1-14
        scatter_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            if group_out != group_in:
                group_flux = flux[group_out]
                for i in range(self.__flux_row):
                    for j in range(self.__flux_col):
                        v_i = self.__m2v(i,j)
                        if i==0 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in])
                        # left-top
                        elif i==0 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]) 
                        # right-bottom               
                        elif i==self.__flux_row-1 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in])   
                        # right-top
                        elif i==self.__flux_row-1 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]) 
                        # left
                        elif i==0 and j>0 and j<self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]) 
                        # top
                        elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                    self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in])         
                        # right
                        elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in])    
                        # bottom     
                        elif i>0 and i<self.__flux_row-1 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in])   
                        # interior
                        else:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in])  
        # np.savetxt('scattering_term.asu',scatter_term)
        return scatter_term
    def define_fission_term(self,group_in,flux=None):
        if flux==None:
            flux = self.__flux
        # Validated 2021-1-14
        fission_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            # group_flux = self.__flux[group_out]
            group_flux = flux[group_out]
            for i in range(self.__flux_row):
                for j in range(self.__flux_col):
                    v_i = self.__m2v(i,j)
                    # left-bottom
                    if i==0 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4])
                    # left-top
                    elif i==0 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                                self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]) 
                    # right-bottom               
                    elif i==self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                                self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4])   
                    # right-top
                    elif i==self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                                self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]) 
                    # left
                    elif i==0 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]) 
                    # top
                    elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                                self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4])         
                    # right
                    elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                                self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4])    
                    # bottom     
                    elif i>0 and i<self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4])   
                    # interior
                    else:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                                self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4])  
        return fission_term

    def define_absorp_operator_sparse(self,group):
        # Validated 2021-1-14
        __row = np.zeros(self.__flux_col*self.__flux_row,dtype=np.int32)
        __col = np.zeros(self.__flux_col*self.__flux_row,dtype=np.int32)
        __data = np.zeros(self.__flux_col*self.__flux_row)
        k = 0
        for i in range(self.__flux_row):
            for j in range(self.__flux_col):
                v_i = self.__m2v(i,j)
                __row[k] = v_i
                __col[k] = v_i
                # left-bottom
                if i==0 and j==0:
                    __data[k] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1])
                # left-top
                elif i==0 and j==self.__flux_col-1:
                    __data[k] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                            self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]) 
                # right-bottom               
                elif i==self.__flux_row-1 and j==0:
                    __data[k] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                            self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1])   
                # right-top
                elif i==self.__flux_row-1 and j==self.__flux_col-1:
                    __data[k] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                            self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]) 
                # left
                elif i==0 and j>0 and j<self.__flux_col-1:
                    __data[k] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]) 
                # top
                elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                    __data[k] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                            self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1])         
                # right
                elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                    __data[k] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1])    
                # bottom     
                elif i>0 and i<self.__flux_row-1 and j==0:
                    __data[k] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1])   
                # interior
                else:
                    __data[k] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]) 
                k += 1  
        # np.savetxt('absorp_op.asu',absorp_op)
        absorp_op = csr_matrix((__data, (__row, __col)), shape=(self.__flux_col*self.__flux_row, self.__flux_col*self.__flux_row))
        return absorp_op

    def define_diffus_operator_sparse(self,group):
        # Validated 2021-1-14
        __row = np.zeros(5*self.__flux_col*self.__flux_row-2*self.__flux_col-2*self.__flux_row,dtype=np.int32)
        __col = np.zeros(5*self.__flux_col*self.__flux_row-2*self.__flux_col-2*self.__flux_row,dtype=np.int32)
        __data = np.zeros(5*self.__flux_col*self.__flux_row-2*self.__flux_col-2*self.__flux_row)
        k = 0
        for i in range(self.__flux_row):
            for j in range(self.__flux_col):
                v_i = self.__m2v(i,j)
                v_i_left = self.__m2v(i-1,j)
                v_i_right = self.__m2v(i+1,j)
                v_i_top = self.__m2v(i,j+1)
                v_i_bottom = self.__m2v(i,j-1)

                # left-bottom
                if i==0 and j==0:
                    #
                    __row[k],__col[k] = v_i,v_i
                    __data[k] += -0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_x[i]+\
                    0.5*(1-self.__boundary_beta['left'])/(1+self.__boundary_beta['left']))*self.geo_m_y[j]
                    __data[k] += -0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_y[j]+\
                        0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom']))*self.geo_m_x[i]
                    k += 1
                    #
                    __row[k],__col[k] = v_i,v_i_right
                    __data[k] += 0.5*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_x[i]
                    k += 1
                    #
                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 0.5*self.geo_m_x[i]*self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_y[j]
                    k += 1
                # left-top
                elif i==0 and j==self.__flux_col-1:
                    __row[k],__col[k] = v_i,v_i
                    __data[k] += -2*0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]/self.geo_m_x[i]+\
                        0.5*(1-self.__boundary_beta['left'])/(1+self.__boundary_beta['left']))*self.geo_m_y[j-1]
                    __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*\
                        self.geo_m_x[i]/self.geo_m_y[j-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_right    
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]/self.geo_m_y[j-1]
                    k += 1
                # right-bottom
                elif i==self.__flux_row-1 and j==0:
                    __row[k],__col[k] = v_i,v_i
                    __data[k] += -2*0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]/self.geo_m_y[j]+\
                        0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom']))*self.geo_m_x[i-1]
                    __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*\
                        self.geo_m_y[j]/self.geo_m_x[i-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i-1]
                    k += 1
                # right-top
                elif i==self.__flux_row-1 and j==self.__flux_col-1:
                    __row[k],__col[k] = v_i,v_i
                    __data[k] += -4*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*\
                        self.geo_m_x[i-1]/self.geo_m_y[j-1]
                    __data[k] += -4*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*\
                        self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    k += 1
                    
                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j-1]
                    k += 1

                # left
                elif i==0 and j>0 and j<self.__flux_col-1:
                    __row[k],__col[k] = v_i,v_i
                    for index_c in range(2):
                        index_r = 0
                        __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]-\
                                0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom'])*0.5*self.geo_m_y[j-index_c]
                    for index_c in range(2):
                        index_r = 0
                        __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_right
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i]
                    k += 1
                    
                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]/self.geo_m_y[j]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]/self.geo_m_y[j-1]        
                    k += 1        

                # bottom
                elif j==0 and i>0 and i<self.__flux_row-1:
                    __row[k],__col[k] = v_i,v_i
                    for index_r in range(2):
                        index_c = 0
                        __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]-\
                                0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom'])*0.5*self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        index_c = 0
                        __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    k += 1
                    
                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j]
                    k += 1
                    
                    __row[k],__col[k] = v_i,v_i_right
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += 0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i-1] 
                    k += 1
                
                # top
                elif j==self.__flux_col-1 and i>0 and i<self.__flux_row-1:
                    __row[k],__col[k] = v_i,v_i
                    for index_r in range(2):
                        index_c = 1
                        __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        index_c = 1
                        __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_right
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j-1]
                    k += 1

                # right
                elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                    __row[k],__col[k] = v_i,v_i
                    for index_c in range(2):
                        index_r = 1
                        __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_c in range(2):
                        index_r = 1
                        __data[k] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i-1]
                    k += 1
                    
                # Interior
                else:
                    __row[k],__col[k] = v_i,v_i
                    for index_r in range(2):
                        for index_c in range(2):
                            __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                                self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        for index_c in range(2):
                            __data[k] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                                self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_right
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_left
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i-1]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_top
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j]
                    k += 1

                    __row[k],__col[k] = v_i,v_i_bottom
                    __data[k] += 0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j-1]
                    k += 1
        # np.savetxt('diffus_op.asu',diffus_op)
        diffus_op = csr_matrix((__data, (__row, __col)), shape=(self.__flux_col*self.__flux_row, self.__flux_col*self.__flux_row))
        return diffus_op

    def define_Q_term(self,flux=None):
        if flux==None:
            flux = self.__flux
        # Validated 2021-1-14
        fission_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            # group_flux = self.__flux[group_out]
            group_flux = flux[group_out]
            for i in range(self.__flux_row):
                for j in range(self.__flux_col):
                    v_i = self.__m2v(i,j)
                    # left-bottom
                    if i==0 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                                self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2])
                    # left-top
                    elif i==0 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                            self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                                self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]) 
                    # right-bottom               
                    elif i==self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                            self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                                self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2])   
                    # right-top
                    elif i==self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                                self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]) 
                    # left
                    elif i==0 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                        self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                                self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]) 
                    # top
                    elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                                self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2])         
                    # right
                    elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                                self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2])    
                    # bottom     
                    elif i>0 and i<self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                                self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2])   
                    # interior
                    else:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]+\
                            self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]+\
                                self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2])  
        return fission_term

    # Adjoint operator (scattering and nu-fission)
    def define_adjoint_scattering_term(self,group_out,flux=None):
        if flux==None:
            flux = self.__flux_adjoint
        # Adjoint scattering term 
        scatter_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_in in range(self.__group_num):
            if group_in != group_out:
                group_flux = flux[group_in]
                for i in range(self.__flux_row):
                    for j in range(self.__flux_col):
                        v_i = self.__m2v(i,j)
                        if i==0 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in])
                        # left-top
                        elif i==0 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]) 
                        # right-bottom               
                        elif i==self.__flux_row-1 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in])   
                        # right-top
                        elif i==self.__flux_row-1 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]) 
                        # left
                        elif i==0 and j>0 and j<self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]) 
                        # top
                        elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                    self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in])         
                        # right
                        elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in]+\
                                self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in])    
                        # bottom     
                        elif i>0 and i<self.__flux_row-1 and j==0:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                    self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in])   
                        # interior
                        else:
                            scatter_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][3][group_in]+\
                            self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][3][group_in]+\
                                self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][3][group_in]+\
                                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][3][group_in])  
        # np.savetxt('scattering_term.asu',scatter_term)
        return scatter_term
    
    def define_adjoint_fission_term(self,group_out,flux=None):
        if flux==None:
            flux = self.__flux_adjoint
        # Validated 2021-1-14
        fission_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_in in range(self.__group_num):
            # group_flux = self.__flux[group_out]
            group_flux = flux[group_in]
            for i in range(self.__flux_row):
                for j in range(self.__flux_col):
                    v_i = self.__m2v(i,j)
                    # left-bottom
                    if i==0 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4])
                    # left-top
                    elif i==0 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                                self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]) 
                    # right-bottom               
                    elif i==self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                                self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4])   
                    # right-top
                    elif i==self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                                self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]) 
                    # left
                    elif i==0 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]) 
                    # top
                    elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                                self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4])         
                    # right
                    elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                                self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4])    
                    # bottom     
                    elif i>0 and i<self.__flux_row-1 and j==0:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                                self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4])   
                    # interior
                    else:
                        fission_term[v_i] += 0.25*group_flux[v_i]*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j]][group_in][4]+\
                        self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group_out][2]*self.__XS_ls[self.mat_m[i,j]][group_in][4]+\
                            self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i,j-1]][group_in][4]+\
                                self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group_out][2]*self.__XS_ls[self.mat_m[i-1,j-1]][group_in][4])  
        return fission_term
    
    def __define_matrix(self):
        self.__matrix_LU = [None]*self.__group_num
        for group in range(self.__group_num):
            self.__matrix_A[group] = -self.define_diffus_operator_sparse(group)+\
                self.define_absorp_operator_sparse(group)
            self.__matrix_LU[group] = splu(self.__matrix_A[group])
    
    # def __define_source(self,k):
    #     # Add the scattering source to the fission source
    #     for group in range(self.__group_num):
    #         self.__source_b[group] = self.define_scattering_term(group)
    #         self.__source_b[group] += self.__fission_source[group]/k

    def __define_source_group(self,k,group,external_source=0,source_type='Normal'):
        if source_type == 'Normal':
            self.__source_b[group] = self.define_scattering_term(group)+\
                self.__fission_source[group]/k+external_source
        elif source_type == 'Adjoint':
            self.__source_b[group] = self.define_adjoint_scattering_term(group)+\
                self.__fission_source[group]/k+external_source
    
    def __define_fission_source(self,source_type='Normal'):
        # Initial or update the fission source
        if source_type == 'Normal':
            for group in range(self.__group_num):
                self.__fission_source[group] = self.define_fission_term(group)
        elif source_type == 'Adjoint':
            for group in range(self.__group_num):
                self.__fission_source[group] = self.define_adjoint_fission_term(group,flux=self.__flux_adjoint)
        
    def __flux2Q_pointwise(self,flux):
        # Q distribution (None group)
        # Q_dist = 0
        # for group in range(self.__group_num):
        #     Q_dist += self.define_fission_term(group,flux=flux)
        # return Q_dist/self.__mesh_area
        Q_dist = self.define_Q_term(flux=flux)
        return Q_dist/self.__mesh_area

    def __calculate_total_Q(self,fission_dist,dist_type):
        if dist_type == 'blockwise':
            temp_dist = fission_dist*self.__block_area
        elif dist_type == 'pointwise':
            temp_dist = fission_dist*self.__mesh_area
        return temp_dist.sum()  

    # Not recommended
    def __calculate_fission(self,source_type='Normal'):
        # self.__fission_dist_last = self.__fission_dist
        # self.__fission_dist = 0
        # for group in range(self.__group_num):
        #     self.__fission_dist += self.__fission_source[group]
        if source_type=='Normal':
            self.__fission_dist_last = self.__fission_dist
            self.__fission_dist = self.define_Q_term()
            return self.__fission_dist.sum()  
        elif source_type=='Adjoint':
            self.__fission_dist_last = self.__fission_dist
            self.__fission_dist = 0
            for group in range(self.__group_num):
                self.__fission_dist += self.__fission_source[group]
            return self.__calculate_total_Q(self.__fission_dist,'pointwise')
        
    def __calculate_dist_pointwise(self,dist):
        return dist/self.__mesh_area

    def __get_dist_error(self,dist,dist_last):
        disp = abs((dist-dist_last)/dist)
        return np.nanmax(disp)

    def __calculate_fission_dist_error(self):
        return self.__get_dist_error(self.__fission_dist,self.__fission_dist_last)

    def __pointwise2blockwise(self,pointwise_dist):
        blockwise_dist = np.zeros((self.__block_row,self.__block_col))
        pointwise_dist_new = pointwise_dist*self.__mesh_area
        pointwise_dist_new = np.reshape(pointwise_dist_new,(self.__flux_row,self.__flux_col),order='F')
        for j in range(self.__block_col):
            for i in range(self.__block_row):
                blockwise_dist[i,j] = pointwise_dist_new[self.__x_mat_index[i]:(self.__x_mat_index[i+1]+1),
                self.__y_mat_index[j]:(self.__y_mat_index[j+1]+1)].sum()
        blockwise_dist = blockwise_dist.flatten(order='F')
        return blockwise_dist/self.__block_area
    
    def __blockwise2pointwise(self,blockwise_dist):
        pointwise_dist = np.zeros((self.__flux_row,self.__flux_col))
        for j in range(self.__block_col):
            for i in range(self.__block_row):
                v_i = self.__m2v_block(i,j)
                pointwise_dist[self.__x_mat_index[i]:(self.__x_mat_index[i+1]+1),
                self.__y_mat_index[j]:(self.__y_mat_index[j+1]+1)] = blockwise_dist[v_i]
        pointwise_dist = pointwise_dist.flatten(order='F')
        return pointwise_dist

    def __save_mat(self,time_index,save_type='SCM',file_name='transient_results'):
        self.__results = {}  
        if save_type=='SCM' or save_type=='SCM_adaptive':
            if time_index == 0:
                self.__results_Q[time_index] = 1
                self.__results_amp_freq[time_index] = 0.0
            else:
                self.__results_Q[time_index] = self.__real_total_Q
                self.__results_amp_freq[time_index] = self.__amp_freq
            
            for i in range(self.__group_num):
                self.__results_shape_freq[i,time_index,:] = self.__shape_freq[i]
            for i in range(self.__delay_group_num):
                self.__results_precursor_conc[i,time_index,:] = self.__precursor_conc[i]
                self.__results_precursor_freq[i,time_index,:] = self.__precursor_freq[i]
            
            # if save_type =='SCM':
            #     self.__results['Time steps'] = self.__time_steps
            #     self.__results['Relative power'] = self.__results_Q
            #     self.__results['Amplitude frequency'] = self.__results_amp_freq
            #     self.__results['Shape frequency'] = self.__results_shape_freq
            #     self.__results['Precursor concentration'] = self.__results_precursor_conc
            #     self.__results['Precursor frequency'] = self.__results_precursor_freq
            self.__results['Time steps'] = self.__time_steps[0:(time_index+1)]
            self.__results['Relative power'] = self.__results_Q[0:(time_index+1)]
            self.__results['Amplitude frequency'] = self.__results_amp_freq[0:(time_index+1)]
            self.__results['Shape frequency'] = self.__results_shape_freq[:,0:(time_index+1),:]
            self.__results['Precursor concentration'] = self.__results_precursor_conc[:,0:(time_index+1),:]
            self.__results['Precursor frequency'] = self.__results_precursor_freq[:,0:(time_index+1),:]

            # sio.savemat('{}/{}.mat'.format(self.__folder_name,file_name),self.__results)
            np.save('{}/{}.npy'.format(self.__folder_name,file_name),self.__results)


    def __save_result(self,name='time_step_0',save_type='steady'):
        self.__get_ordinate()
        if save_type=='steady':
            # Save the flux distribution and the material layout to .vtk
            # Please use ParaView or etc. to visualize it
            self.__fission_dist_pointwise = self.__calculate_dist_pointwise(self.__fission_dist)
            ####Test#####
            self.__fission_dist_blockwise = self.__pointwise2blockwise(self.__fission_dist_pointwise)

            # Normalized
            total_Q = self.__calculate_total_Q(self.__fission_dist_pointwise,dist_type='pointwise')
            self.__real_total_Q = 1.0
            normal_c = self.__real_total_Q/total_Q
            self.__fission_dist_pointwise = normal_c*self.__fission_dist_pointwise
            self.__fission_dist_blockwise = normal_c*self.__fission_dist_blockwise
            for group in range(self.__group_num):
                self.__flux[group] = normal_c*self.__flux[group]*self.__k_init
            print('Total Q (initial): {}'.format(1))

        if save_type=='steady' or save_type=='transient':    
            fission_dist_blockwise = np.reshape(self.__fission_dist_blockwise,(self.__block_row,self.__block_col),order='F') 
            np.savetxt('{}/fission_blockwise_{}.asu'.format(self.__folder_name,name),fission_dist_blockwise,fmt="%1.4e")
        #############
        with open('{}/{}.vtk'.format(self.__folder_name,name),'w') as file_obj:
            file_obj.write('# vtk DataFile Version 2.0'+'\n')
            file_obj.write('FDM Code for neutron diffusion'+'\n')
            file_obj.write('ASCII'+'\n')
            file_obj.write('DATASET RECTILINEAR_GRID'+'\n')
            file_obj.write('DIMENSIONS {} {} {}'.format(self.__flux_row,self.__flux_col,1)+'\n')
            file_obj.write('X_COORDINATES {} float'.format(self.__flux_row)+'\n')
            file_obj.write(' '.join(str(i) for i in self.__x_ordinate)+'\n')
            file_obj.write('Y_COORDINATES {} float'.format(self.__flux_col)+'\n')
            file_obj.write(' '.join(str(i) for i in self.__y_ordinate)+'\n')
            file_obj.write('Z_COORDINATES 1 float'+'\n')
            file_obj.write('0.0'+'\n')
            file_obj.write('POINT_DATA {}'.format(self.__flux_row*self.__flux_col)+'\n')

            if save_type=='steady' or save_type=='transient':
                # Flux
                for group in range(self.__group_num):
                    flux = self.__flux[group]
                    file_obj.write('SCALARS Flux_group_{} float 1'.format(group)+'\n')
                    file_obj.write('LOOKUP_TABLE default'+'\n')
                    for i in range(self.__flux_row*self.__flux_col):
                        file_obj.write('{}'.format(flux[i])+'\n')
                # Fission
                file_obj.write('SCALARS Fission_source float 1'+'\n')
                file_obj.write('LOOKUP_TABLE default'+'\n')
                for i in range(self.__flux_row*self.__flux_col):
                    file_obj.write('{}'.format(self.__fission_dist_pointwise[i])+'\n') 
            elif save_type=='adjoint':
                # Adjoint flux
                for group in range(self.__group_num):
                    flux = self.__flux_adjoint[group]
                    file_obj.write('SCALARS Adjoint_flux_group_{} float 1'.format(group)+'\n')
                    file_obj.write('LOOKUP_TABLE default'+'\n')
                    for i in range(self.__flux_row*self.__flux_col):
                        file_obj.write('{}'.format(flux[i])+'\n')
            # Material
            file_obj.write('CELL_DATA {}'.format(self.__mat_row*self.__mat_col)+'\n')
            file_obj.write('SCALARS Material_layout int 1'+'\n')
            file_obj.write('LOOKUP_TABLE default'+'\n')
            for j in range(self.__mat_col):
                for i in range(self.__mat_row):
                    file_obj.write('{}'.format(self.mat_m[i,j])+'\n')
          
    def __get_ordinate(self):
        # Get ordinates of rectilinear grid
        self.__x_ordinate = np.zeros(self.__flux_row)
        self.__y_ordinate = np.zeros(self.__flux_col)
        for i in range(self.__flux_row-1):
            self.__x_ordinate[i+1] = self.__x_ordinate[i]+self.geo_m_x[i]
        for i in range(self.__flux_col-1):
            self.__y_ordinate[i+1] = self.__y_ordinate[i]+self.geo_m_y[i]


    def solve_source_iter_correct(self,max_iter,k_tolerance=1e-5,flux_tolerance=1e-3,initial_k=1.0):
        # With an acceleration trick
        k = initial_k
        self.__define_matrix()
        self.__initial_flux()
        print('####################################')
        print('Timestep {} calculation begins'.format(0))
        for i in range(max_iter):
            # Update source
            if i==0:
                self.__define_fission_source()
                last_fission = self.__calculate_fission()
            # Solve flux
            # Ax = b
            for group in range(self.__group_num):
                self.__define_source_group(k,group)
                # self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
                self.__flux[group] = self.__matrix_LU[group].solve(self.__source_b[group])
            # Update k and source
            self.__define_fission_source()
            new_fission = self.__calculate_fission()
            new_k = k*new_fission/last_fission
            # Stopping criterion
            fission_dist_error = self.__calculate_fission_dist_error()
            self.__k = new_k
            self.__k_init = new_k
            if abs(k-new_k)/new_k<=k_tolerance and fission_dist_error<=flux_tolerance:
                print('Iteration {}: Met the criteria and k = {}, εq = {}'.format(i+1,new_k,fission_dist_error))
                self.__save_result()
                break 
            else:
                print('Iteration {}: Eigenvalue k = {}'.format(i+1,new_k))
                # print('Error of fission source εq = {}'.format(fission_dist_error))
            if i==max_iter-1:
                print('Reached the maximum iteration number{}'.format(i+1))
                self.__save_result()
            # Update others
            last_fission = new_fission
            k = new_k

    def solve_source_adjoint_iter_correct(self,max_iter,k_tolerance=1e-5,flux_tolerance=1e-3,initial_k=1.0):
        # With an acceleration trick
        k = initial_k
        self.__define_matrix()
        self.__initial_flux(flux_type='Adjoint')
        print('####################################')
        print('Adjoint equations calculation begins')
        for i in range(max_iter):
            # Update source
            if i==0:
                self.__define_fission_source(source_type='Adjoint')
                last_fission = self.__calculate_fission(source_type='Adjoint')
            # Solve flux
            # Ax = b
            for group in range(self.__group_num):
                self.__define_source_group(k,group,external_source=0,source_type='Adjoint')
                # self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
                self.__flux_adjoint[group] = self.__matrix_LU[group].solve(self.__source_b[group])
            # Update k and source
            self.__define_fission_source(source_type='Adjoint')
            new_fission = self.__calculate_fission(source_type='Adjoint')
            new_k = k*new_fission/last_fission
            # Stopping criterion
            fission_dist_error = self.__calculate_fission_dist_error()
            # self.__k = new_k
            # self.__k_init = new_k
            if abs(k-new_k)/new_k<=k_tolerance and fission_dist_error<=flux_tolerance:
                print('Iteration {}: Met the criteria and k = {}, εq = {}'.format(i+1,new_k,fission_dist_error))
                self.__save_result(name='Initial_adjoint_flux',save_type='adjoint')
                break 
            else:
                print('Iteration {}: Eigenvalue k = {}'.format(i+1,new_k))
                # print('Error of fission source εq = {}'.format(fission_dist_error))
            if i==max_iter-1:
                print('Reached the maximum iteration number{}'.format(i+1))
                self.__save_result(name='Initial_adjoint_flux',save_type='adjoint')
            # Update others
            last_fission = new_fission
            k = new_k


    def solve_transient_SCM_2(self,time_index,time_interval,max_iter,k_tolerance=1e-5,flux_tolerance=1e-3,k_outer_tolerance=1e-6,call_type='Normal'):
        # Ah~
        # Stiffness confinement method for transient problem
        print('####################################')
        print('Timestep {} calculation begins'.format(time_index))
        # First, store some quantities from the last time step 
        # Neutron flux and fission distribution
        flux_last = copy.deepcopy(self.__flux)
        flux_blockwise_last = [None]*self.__group_num
        for group in range(self.__group_num):
            flux_blockwise_last[group] = self.__pointwise2blockwise(flux_last[group])
        fission_dist_pointwise_last = self.__fission_dist_pointwise.copy()
        fission_dist_blockwise_last = self.__fission_dist_blockwise.copy()
        # Frequency and concentration
        amp_freq_last = self.__amp_freq
        shape_freq_last = copy.deepcopy(self.__shape_freq)
        # precursor_freq_last = self.__precursor_freq
        precursor_conc_last = copy.deepcopy(self.__precursor_conc)

        # Amp freq predictor
        if call_type == 'Normal':
            self.__amp_freq_predict = amp_freq_last
        
        self.__amp_freq = self.__amp_freq_predict
        # Pre-update
        if call_type == 'Predict':
            # Calculate real flux and fission source
            real_flux = self.__update_flux_real(flux_pointwise=flux_last,amp=self.__amp_freq,amp_last=amp_freq_last,time_interval=time_interval)
            real_fission_dist_pointwise = self.__flux2Q_pointwise(flux=real_flux)
            real_fission_dist_blockwise = self.__pointwise2blockwise(real_fission_dist_pointwise)
            real_total_Q = self.__calculate_total_Q(fission_dist=real_fission_dist_pointwise,dist_type='pointwise')

            # Update precursor frequency and concentration
            self.__update_conc_exp(flux_blockwise_last,precursor_conc_last,self.__amp_freq,amp_freq_last,self.__shape_freq,shape_freq_last,time_interval)
            self.__update_precursor_freq(precursor_conc=self.__precursor_conc,fission_blockwise=real_fission_dist_blockwise)

        # If the call type is 'Predictor'
        # self.__amp_freq_predict will be determined by the PKE solver


        # Power
        total_Q_normal = self.__calculate_total_Q(fission_dist=fission_dist_pointwise_last,dist_type='pointwise')
        print('Total Q (last): {}'.format(total_Q_normal))
        # total_Q_last = total_Q_normal
        kD_last = 1
        for i in range(max_iter):
            # if i==0:
            #     self.__amp_freq = self.__amp_freq_predict
            # elif i==1:
            #     self.__amp_freq_last = self.__amp_freq_predict
            #     if abs(self.__amp_freq_predict)>=1e-7:
            #         self.__amp_freq = 1.1*self.__amp_freq_predict
            #     else:
            #         self.__amp_freq = self.__amp_freq_predict + 0.1
            # Solve within-group equations 
            self.__update_dynamics_XS()
            self.__define_matrix()
            self.__define_fission_source()
            k = kD_last
            print('####################################')
            print('Outer iteration {} begins'.format(i+1))

            #TODO
            #For test
            # print('Shape frequency:')
            # print(self.__shape_freq)
            for j in range(max_iter):
                if j==0:
                    last_fission = self.__calculate_fission()
                # Solve flux
                # Ax = b
                for group in range(self.__group_num):
                    self.__define_source_group(k,group)
                    # self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
                    self.__flux[group] = self.__matrix_LU[group].solve(self.__source_b[group])
                # Update k and source
                self.__define_fission_source()
                new_fission = self.__calculate_fission()
                new_k = k*new_fission/last_fission
                if abs((k-new_k)/new_k)<=k_tolerance:
                    print('Iteration {}: Met the criteria and k = {}'.format(j+1,new_k))
                    break 
                else:
                    print('Iteration {}: Eigenvalue k = {}'.format(j+1,new_k))
                if j==max_iter-1:
                    print('Reached the maximum iteration number{}'.format(j+1))
                # Update others
                last_fission = new_fission
                k = new_k

            # if i==0:
            #     kD = new_k
            #     print('Outer Iteration {}： kD={}'.format(i+1,kD))
            #     kD_last = kD
            #     continue

            # Update fission source
            fission_dist_pointwise = self.__flux2Q_pointwise(flux=self.__flux)
            total_Q = self.__calculate_total_Q(fission_dist=fission_dist_pointwise,dist_type='pointwise')
            # kD = total_Q/total_Q_last
            kD = new_k
            print('Outer Iteration {}： kD={}'.format(i+1,kD))
            
            # Update amplitude frequency ωT
            # self.__update_amp_freq(kD=kD,fission_dist_pointwise=fission_dist_pointwise,flux_pointwise=self.__flux)
            self.__update_amp_freq_secant(kD=kD,kD_last=kD_last,amp_freq=self.__amp_freq,amp_freq_last=self.__amp_freq_last,iter_index=i)
            kD_last = kD
            # Normalized factor 
            normal_C = total_Q_normal/total_Q
            normal_flux = [None]*self.__group_num
            normal_flux_blockwise = [None]*self.__group_num
            for group in range(self.__group_num):
                normal_flux[group] = normal_C*self.__flux[group]
                normal_flux_blockwise[group] = self.__pointwise2blockwise(normal_flux[group])
            # normal_fission_dist_pointwise = normal_C*fission_dist_pointwise
            # normal_fission_dist_blockwise = self.__pointwise2blockwise(normal_fission_dist_pointwise)
            # total_Q_last = total_Q*normal_C

            # For test
            # TODO
            # print('Flux (last time step):')
            # print(flux_blockwise_last)
            # print('Flux (normalized):')
            # print(normal_flux_blockwise)

            # Update shape frequency
            self.__update_shape_freq(flux_blockwise=normal_flux_blockwise,flux_blockwise_last=flux_blockwise_last,shape_freq_last=shape_freq_last,time_interval=time_interval)

            # Calculate real flux and fission source
            real_flux = self.__update_flux_real(flux_pointwise=normal_flux,amp=self.__amp_freq,amp_last=amp_freq_last,time_interval=time_interval)
            real_fission_dist_pointwise = self.__flux2Q_pointwise(flux=real_flux)
            real_fission_dist_blockwise = self.__pointwise2blockwise(real_fission_dist_pointwise)
            real_total_Q = self.__calculate_total_Q(fission_dist=real_fission_dist_pointwise,dist_type='pointwise')
            print('Total Q: {}'.format(real_total_Q))

            # Update precursor frequency and concentration
            # self.__update_conc(fission_blockwise=real_fission_dist_blockwise,fission_blockwise_last=fission_dist_blockwise_last,precursor_conc_last=precursor_conc_last,time_interval=time_interval)
            self.__update_conc_exp(flux_blockwise_last,precursor_conc_last,self.__amp_freq,amp_freq_last,self.__shape_freq,shape_freq_last,time_interval)
            self.__update_precursor_freq(precursor_conc=self.__precursor_conc,fission_blockwise=real_fission_dist_blockwise)
            # Stopping criterion
            if abs(kD-1)<=k_outer_tolerance and i>0:
                print('Reached the stopping criterion')
                break          
            self.__flux = copy.deepcopy(normal_flux)

        # Save results
        print('Total Q (convergent): {}'.format(real_total_Q))
        self.__real_total_Q = real_total_Q
        self.__flux = copy.deepcopy(real_flux)
        self.__fission_dist_pointwise = real_fission_dist_pointwise
        self.__fission_dist_blockwise = real_fission_dist_blockwise
        if self.__vtk_save:
            self.__save_result(name='time_step_{}'.format(time_index),save_type='transient')
        self.__save_mat(time_index=time_index)
        self.__k = kD



    def solve_transient_SCM_PKE_predictor(self,time_index,time_interval,max_iter,k_tolerance=1e-5,flux_tolerance=1e-3,k_outer_tolerance=1e-6):
        # TODO
        if time_index==1:
            self.__n_PKE = np.zeros(1+self.__delay_group_num)

        # Predict reactivity with new XS information
        time_XS = self.__narray_XS()
        real_flux_blockwise = [None]*self.__group_num
        adjoint_flux_blockwise = [None]*self.__group_num
        for group in range(self.__group_num):
            real_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux[group])
            adjoint_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux_adjoint[group])
        predicted_reactivity = self.__get_reactivity(self.__fission_dist_blockwise,real_flux_blockwise,adjoint_flux_blockwise,time_XS,self.__init_XS_narray)
        print('Predicted reactivity: {}'.format(predicted_reactivity))

        # PKE predictor
        time_start = self.__time_steps[time_index-1]
        time_end = self.__time_steps[time_index]

        # Get dynamics parameters
        dynamics_parameters = self.__get_dynamics_parameters(time_index,self.__fission_dist_blockwise,real_flux_blockwise,adjoint_flux_blockwise,self.__precursor_conc,time_XS,self.__results_reactivity[time_index-1])

        # print('Dynamics parameters:')
        # print(dynamics_parameters)

        # Initilize predicted PKE solver
        PKE = solver_PKE(delay_group_num=self.__delay_group_num,time_start=time_start,time_end=time_end)
        PKE.initial_dynamics_parameters(dynamics_parameters['neutron_time'],dynamics_parameters['delay_fractions'],dynamics_parameters['decay_constants'])
        PKE.initial_density(neutron_density=dynamics_parameters['neutron_density'],precursor_conc=dynamics_parameters['precursor_conc'])
        PKE.time_variant_reactivity(init_rho=self.__results_reactivity[time_index-1],end_rho=predicted_reactivity,variant_type='linear')
        results = PKE.predictor_solve()
        omega = results['omega']
        self.__amp_freq_predict = 2*omega-self.__amp_freq
        print('Predicted amplitude frequency: {}'.format(self.__amp_freq_predict))

        # SCM
        self.solve_transient_SCM_2(time_index,time_interval,max_iter,k_tolerance,flux_tolerance,k_outer_tolerance,'Predictor')

        # Actual reactivity
        time_XS = self.__narray_XS()
        real_flux_blockwise = [None]*self.__group_num
        adjoint_flux_blockwise = [None]*self.__group_num
        for group in range(self.__group_num):
            real_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux[group])
            adjoint_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux_adjoint[group])
        reactivity = self.__get_reactivity(self.__fission_dist_blockwise,real_flux_blockwise,adjoint_flux_blockwise,time_XS,self.__init_XS_narray)
        self.__results_reactivity[time_index] = reactivity
        print('Reactivity: {}'.format(reactivity))

    def evaluate_adaptive_time_step(self,time_index,pre_time_interval,error_tolerance):
        # time_interval  = self.__max_time_step
        time_interval = pre_time_interval
        # TODO
        if time_index==1:
            self.__n_PKE = np.zeros(1+self.__delay_group_num)

        # Predict reactivity with new XS information
        time_XS = self.__narray_XS()
        real_flux_blockwise = [None]*self.__group_num
        adjoint_flux_blockwise = [None]*self.__group_num
        for group in range(self.__group_num):
            real_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux[group])
            adjoint_flux_blockwise[group] = self.__pointwise2blockwise(self.__flux_adjoint[group])
        predicted_reactivity = self.__get_reactivity(self.__fission_dist_blockwise,real_flux_blockwise,adjoint_flux_blockwise,time_XS,self.__init_XS_narray)
        print('Predicted reactivity (1st): {}'.format(predicted_reactivity))

        # PKE predictor
        time_start = self.__time_steps[time_index-1]
        time_end = time_start+time_interval

        # Get dynamics parameters
        dynamics_parameters = self.__get_dynamics_parameters(time_index,self.__fission_dist_blockwise,real_flux_blockwise,adjoint_flux_blockwise,self.__precursor_conc,time_XS,self.__results_reactivity[time_index-1])

        # Initilize predicted PKE solver
        PKE = solver_PKE(delay_group_num=self.__delay_group_num,time_start=time_start,time_end=time_end)
        PKE.initial_dynamics_parameters(dynamics_parameters['neutron_time'],dynamics_parameters['delay_fractions'],dynamics_parameters['decay_constants'])
        PKE.initial_density(neutron_density=dynamics_parameters['neutron_density'],precursor_conc=dynamics_parameters['precursor_conc'])
        PKE.time_variant_reactivity(init_rho=self.__results_reactivity[time_index-1],end_rho=predicted_reactivity,variant_type='linear')
        results = PKE.predictor_solve(time_eval=True)
        omega = results['omega']
        omega_dif_2 = results['omega_dif_2'] # For error evaluation
        self.__amp_freq_predict = 2*omega-self.__amp_freq
        print('Predicted amplitude frequency (1st): {}'.format(self.__amp_freq_predict))

        # Time step select
        time_interval = self.__get_time_step(time_index,time_interval,omega_dif_2,error_tolerance)

        return time_interval        

    def solve_transient_Implicit_Euler(self,time_index,time_interval,max_iter,k_tolerance=1e-5,flux_tolerance=1e-3):
        # Implicit Euler method (order-1)
        # Ah~
        print('####################################')
        print('Timestep {} calculation begins'.format(time_index))
        # First, store some quantities from the last time step 
        # Neutron flux and fission distribution
        flux_last = copy.deepcopy(self.__flux)
        flux_blockwise_last = [None]*self.__group_num
        for group in range(self.__group_num):
            flux_blockwise_last[group] = self.__pointwise2blockwise(flux_last[group])
        fission_dist_pointwise_last = self.__fission_dist_pointwise.copy()
        fission_dist_blockwise_last = self.__fission_dist_blockwise.copy()
        # Precursor concentration
        precursor_conc_last = copy.deepcopy(self.__precursor_conc)
        # Initial dynamics XS
        self.__update_dynamics_XS(transient_algorithm='Implicit_Euler',time_interval=time_interval)
        # Define time source
        self.__define_time_source(flux_blockwise=flux_blockwise_last,precursor_conc_last=precursor_conc_last,time_interval=time_interval,time_index=time_index)
        self.__define_matrix()
        self.__define_fission_source()

        #TODO
        #For test
        fission_dist_pointwise_last_iter = fission_dist_pointwise_last.copy()
        # Solve the fixed source problem
        for i in range(max_iter):
            # Solve flux
            # Ax = b
            for group in range(self.__group_num):
                self.__define_source_group(1,group,self.__time_source[group])
                # self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
                self.__flux[group] = self.__matrix_LU[group].solve(self.__source_b[group])
            # Update source
            self.__define_fission_source()
            # Q
            fission_dist_pointwise = self.__flux2Q_pointwise(flux=self.__flux)
            total_Q = self.__calculate_total_Q(fission_dist=fission_dist_pointwise,dist_type='pointwise')
            # Stopping criterion for fission distribution
            fission_dist_error = self.__get_dist_error(fission_dist_pointwise,fission_dist_pointwise_last_iter)
            fission_dist_pointwise_last_iter = fission_dist_pointwise
            if fission_dist_error<=flux_tolerance:
                print('Iteration {}: Total Q: {}, Met the criteria εq = {}'.format(i+1,total_Q,fission_dist_error))
                break 
            else:
                print('Iteration {}: Total Q: {}, Error of fission source εq = {}'.format(i+1,total_Q,fission_dist_error))
            if i==max_iter-1:
                print('Reached the maximum iteration number{}'.format(i+1))
        # TODO       
        self.__fission_dist_pointwise = fission_dist_pointwise
        self.__fission_dist_blockwise = self.__pointwise2blockwise(self.__fission_dist_pointwise)
        # Update precursor concentrations
        self.__update_conc(fission_blockwise=self.__fission_dist_blockwise,fission_blockwise_last=fission_dist_blockwise_last,precursor_conc_last=precursor_conc_last,time_interval=time_interval)
        # Save results 
        if self.__vtk_save:
            self.__save_result(name='time_step_{}'.format(time_index),save_type='transient')
    # For test
    def get_var(self,var):
        if var == 'amp_freq':
            res = self.__amp_freq
        elif var == 'precursor_conc':
            res = self.__precursor_conc
        elif var == 'shape_freq':
            res = self.__shape_freq
        elif var == 'precursor_freq':
            res = self.__precursor_freq
        elif var == 'reactivity':
            res = self.__results_reactivity
        elif var == 'block_flux':
            flux_blockwise = [None]*self.__group_num
            for group in range(self.__group_num):
                flux_blockwise[group] = self.__pointwise2blockwise(self.__flux[group])
            res = flux_blockwise
        return res

    def get_results(self):
        return self.__results

            
