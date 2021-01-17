import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from collections import Iterable
import os

'''
The finite difference code for the neutron diffusion equation
Wei Xiao
Shanghai Jiao Tong University
School of Nuclear Science and Engineering
bearsanxw@gmail.com
Only for the test of transient algorithms
2021-1-16
'''
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
    
    def update_material(self,mat_id,material):
        if mat_id in self.__XS_ls:
            XS = self.__to_XS(material)
            self.__XS_ls[mat_id] = XS
        else:
            print('There is no material {}'.format(mat_id))
    
    # Material list
    # 1st: material; 2nd: group; 3rd: XS: Dg, Σag, vΣfg, Σs, χ    
    def __to_XS(self,material):
        XS = [None]*self.__group_num
        for group in range(self.__group_num):
            XS_group = [None]*5
            XS_group[0] = material.get_XS('Diffusion_coef')[group]
            XS_group[2] = material.get_XS('XS_nu_fission')[group]
            XS_group[3] = material.get_XS('XS_scatter')[group]
            XS_group[4] = material.get_XS('XS_fission_spectrum')[group]
            for i in range(self.__group_num):
                if i==0:
                    XS_group[1] = material.get_XS('XS_absorption')[group]
                else:
                    XS_group[1] += XS_group[3][i]
            XS[group] = XS_group
        return XS

    def set_geometry(self,geo):

        # Initial row and column number
        # row (x)
        self.__mat_row = 0
        self.__x_mat_index = [0]*(geo.x_block+1)
        for i,x_dim in enumerate(geo.x_dim):
            self.__mat_row += x_dim
            if i==0:
                self.__x_mat_index[i+1] = x_dim
            else:
                self.__x_mat_index[i+1] = x_dim+self.__x_mat_index[i]
        self.__x_mat_index[-1] -= 1
        self.__mat_row -= 1
        
        # column (y)
        self.__mat_col = 0
        self.__y_mat_index = [0]*(geo.y_block+1)
        for i,y_dim in enumerate(geo.y_dim):
            self.__mat_col += y_dim
            if i==0:
                self.__y_mat_index[i+1] = y_dim
            else:
                self.__y_mat_index[i+1] = y_dim+self.__y_mat_index[i]
        self.__y_mat_index[-1] -= 1
        self.__mat_col -= 1

        self.__flux_row = self.__mat_row+1
        self.__flux_col = self.__mat_col+1    

        # Initial material layout
        self.__define_mat_m(geo)
        self.__define_geo_m_x(geo)
        self.__define_geo_m_y(geo)
        np.savetxt('{}/material_layout.asu'.format(self.__folder_name),self.mat_m,fmt="%i")

    def __v2m(self,v_i):
        m_j = v_i//self.__flux_row
        m_i = v_i-m_j*self.__flux_row
        return m_i,m_j
    def __m2v(self,m_i,m_j):
        v_i = m_j*self.__flux_row+m_i
        return v_i

    def __define_mat_m(self,geo):
        self.mat_m = np.zeros((self.__mat_row,self.__mat_col),dtype=np.int32)
        for j in range(geo.y_block):
            for i in range(geo.x_block):
                self.mat_m[self.__x_mat_index[i]:self.__x_mat_index[i+1],
                self.__y_mat_index[j]:self.__y_mat_index[j+1]] = geo.get_block_mat(i,j)

    #TODO
    def __define_geo_m_x(self,geo):
        self.geo_m_x = np.zeros(self.__mat_row)
        for i in range(geo.x_block):
            self.geo_m_x[self.__x_mat_index[i]:self.__x_mat_index[i+1]] = geo.x_size[i]/geo.x_dim[i]
    
    def __define_geo_m_y(self,geo):
        self.geo_m_y = np.zeros(self.__mat_col)
        for i in range(geo.y_block):
            self.geo_m_y[self.__y_mat_index[i]:self.__y_mat_index[i+1]] = geo.y_size[i]/geo.y_dim[i]

    def __initial_flux(self):
        # Initial flux storage
        self.__flux = [None]*self.__group_num
        for i in range(self.__group_num):
            self.__flux[i] = np.ones(self.__flux_col*self.__flux_row)
        self.__fission_dist = np.ones(self.__flux_col*self.__flux_row)
        self.__fission_dist_last = np.ones(self.__flux_col*self.__flux_row)

    def boundary_set(self,loc,boundary_type,beta=0):
        if boundary_type=='reflective':
            self.__boundary[loc]=0
            self.__boundary_beta[loc] = beta
        else:
            self.__boundary[loc]=1

    def print_boundary(self):
        print('Boundary type')
        print(self.__boundary)

    #TODO
    def define_absorp_operator(self,group):
        # Validated 2021-1-14
        absorp_op = np.zeros((self.__flux_row*self.__flux_col,self.__flux_row*self.__flux_col))
        for i in range(self.__flux_row):
            for j in range(self.__flux_col):
                v_i = self.__m2v(i,j)
                # left-bottom
                if i==0 and j==0:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['left']*self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1])
                # left-top
                elif i==0 and j==self.__flux_col-1:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                    self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                            self.__boundary['left']*self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]) 
                # right-bottom               
                elif i==self.__flux_row-1 and j==0:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                            self.__boundary['right']*self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1])   
                # right-top
                elif i==self.__flux_row-1 and j==self.__flux_col-1:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                    self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                            self.__boundary['right']*self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]) 
                # left
                elif i==0 and j>0 and j<self.__flux_col-1:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                        self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['left']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]) 
                # top
                elif i>0 and i<self.__flux_row-1 and j==self.__flux_col-1:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['top']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                            self.__boundary['top']*self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1])         
                # right
                elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1]+\
                        self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                            self.__boundary['right']*self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1])    
                # bottom     
                elif i>0 and i<self.__flux_row-1 and j==0:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.__boundary['bottom']*self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                            self.__boundary['bottom']*self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1])   
                # interior
                else:
                    absorp_op[v_i,v_i] = 0.25*(self.geo_m_x[i-1]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i-1,j]][group][1]+\
                    self.geo_m_x[i]*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][1]+\
                        self.geo_m_x[i]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i,j-1]][group][1]+\
                            self.geo_m_x[i-1]*self.geo_m_y[j-1]*self.__XS_ls[self.mat_m[i-1,j-1]][group][1])   
        np.savetxt('absorp_op.asu',absorp_op)
        return absorp_op
    def define_diffus_operator(self,group):
        # Validated 2021-1-14
        diffus_op = np.zeros((self.__flux_row*self.__flux_col,self.__flux_row*self.__flux_col))
        for i in range(self.__flux_row):
            for j in range(self.__flux_col):
                v_i = self.__m2v(i,j)
                v_i_left = self.__m2v(i-1,j)
                v_i_right = self.__m2v(i+1,j)
                v_i_top = self.__m2v(i,j+1)
                v_i_bottom = self.__m2v(i,j-1)

                # left-bottom
                if i==0 and j==0:
                    diffus_op[v_i,v_i] += -0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_x[i]+\
                    0.5*(1-self.__boundary_beta['left'])/(1+self.__boundary_beta['left']))*self.geo_m_y[j]
                    diffus_op[v_i,v_i] += -0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_y[j]+\
                        0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom']))*self.geo_m_x[i]
                    diffus_op[v_i,v_i_right] += 0.5*self.geo_m_y[j]*self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_x[i]
                    diffus_op[v_i,v_i_top] += 0.5*self.geo_m_x[i]*self.__XS_ls[self.mat_m[i,j]][group][0]/self.geo_m_y[j]
                # left-top
                elif i==0 and j==self.__flux_col-1:
                    diffus_op[v_i,v_i] += -2*0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]/self.geo_m_x[i]+\
                        0.5*(1-self.__boundary_beta['left'])/(1+self.__boundary_beta['left']))*self.geo_m_y[j-1]
                    diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*\
                        self.geo_m_x[i]/self.geo_m_y[j-1]
                    diffus_op[v_i,v_i_right] += 2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i]
                    diffus_op[v_i,v_i_bottom] += 0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]/self.geo_m_y[j-1]
                # right-bottom
                elif i==self.__flux_row-1 and j==0:
                    diffus_op[v_i,v_i] += -2*0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]/self.geo_m_y[j]+\
                        0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom']))*self.geo_m_x[i-1]
                    diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*\
                        self.geo_m_y[j]/self.geo_m_x[i-1]
                    diffus_op[v_i,v_i_top] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j]
                    diffus_op[v_i,v_i_left] += 0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i-1]
                # right-top
                elif i==self.__flux_row-1 and j==self.__flux_col-1:
                    diffus_op[v_i,v_i] += -4*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*\
                        self.geo_m_x[i-1]/self.geo_m_y[j-1]
                    diffus_op[v_i,v_i] += -4*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*\
                        self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    diffus_op[v_i,v_i_left] += self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    diffus_op[v_i,v_i_bottom] += self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j-1]

                # left
                elif i==0 and j>0 and j<self.__flux_col-1:
                    for index_c in range(2):
                        index_r = 0
                        diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]-\
                                0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom'])*0.5*self.geo_m_y[j-index_c]
                    for index_c in range(2):
                        index_r = 0
                        diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    diffus_op[v_i,v_i_right] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i]
                    diffus_op[v_i,v_i_top] += 0.5*self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]/self.geo_m_y[j]
                    diffus_op[v_i,v_i_bottom] += 0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]/self.geo_m_y[j-1]                

                # bottom
                elif j==0 and i>0 and i<self.__flux_row-1:
                    for index_r in range(2):
                        index_c = 0
                        diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]-\
                                0.5*(1-self.__boundary_beta['bottom'])/(1+self.__boundary_beta['bottom'])*0.5*self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        index_c = 0
                        diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    diffus_op[v_i,v_i_top] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j]
                    diffus_op[v_i,v_i_right] += 0.5*self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i]
                    diffus_op[v_i,v_i_left] += 0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]/self.geo_m_x[i-1] 
                
                # top
                elif j==self.__flux_col-1 and i>0 and i<self.__flux_row-1:
                    for index_r in range(2):
                        index_c = 1
                        diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        index_c = 1
                        diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    diffus_op[v_i,v_i_right] += 2*0.5*self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i]
                    diffus_op[v_i,v_i_left] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1]/self.geo_m_x[i-1]
                    diffus_op[v_i,v_i_bottom] += 0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j-1]

                # right
                elif i==self.__flux_row-1 and j>0 and j<self.__flux_col-1:
                    for index_c in range(2):
                        index_r = 1
                        diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_c in range(2):
                        index_r = 1
                        diffus_op[v_i,v_i] += -2*0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                            self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    diffus_op[v_i,v_i_top] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j]
                    diffus_op[v_i,v_i_bottom] += 2*0.5*self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1]/self.geo_m_y[j-1]
                    diffus_op[v_i,v_i_left] += 0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i-1]


                    
                # Interior
                else:
                    for index_r in range(2):
                        for index_c in range(2):
                            diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                                self.geo_m_y[j-index_c]/self.geo_m_x[i-index_r]
                    for index_r in range(2):
                        for index_c in range(2):
                            diffus_op[v_i,v_i] += -0.5*self.__XS_ls[self.mat_m[i-index_r,j-index_c]][group][0]*\
                                self.geo_m_x[i-index_r]/self.geo_m_y[j-index_c]
                    diffus_op[v_i,v_i_right] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i]
                    diffus_op[v_i,v_i_left] += 0.5*(self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_y[j]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_y[j-1])/self.geo_m_x[i-1]
                    diffus_op[v_i,v_i_top] += 0.5*(self.__XS_ls[self.mat_m[i,j]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j]
                    diffus_op[v_i,v_i_bottom] += 0.5*(self.__XS_ls[self.mat_m[i,j-1]][group][0]*self.geo_m_x[i]+\
                        self.__XS_ls[self.mat_m[i-1,j-1]][group][0]*self.geo_m_x[i-1])/self.geo_m_y[j-1]
        np.savetxt('diffus_op.asu',diffus_op)
        return diffus_op
    def define_scattering_term(self,group_in):
        # Validated 2021-1-14
        scatter_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            if group_out != group_in:
                group_flux = self.__flux[group_out]
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
    def define_fission_term(self,group_in):
        # Validated 2021-1-14
        fission_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            group_flux = self.__flux[group_out]
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

    def __define_matrix(self):
        for group in range(self.__group_num):
            self.__matrix_A[group] = -self.define_diffus_operator_sparse(group)+\
                self.define_absorp_operator_sparse(group)
    
    def __define_source(self,k):
        # Add the scattering source to the fission source
        for group in range(self.__group_num):
            self.__source_b[group] = self.define_scattering_term(group)
            self.__source_b[group] += self.__fission_source[group]/k

    def __define_source_group(self,k,group):
        self.__source_b[group] = self.define_scattering_term(group)+\
            self.__fission_source[group]/k
    
    def __define_fission_source(self):
        # Initial or update the fission source
        for group in range(self.__group_num):
            self.__fission_source[group] = self.define_fission_term(group)

    def __calculate_fission(self):
        self.__fission_dist_last = self.__fission_dist
        self.__fission_dist = 0
        for group in range(self.__group_num):
            self.__fission_dist += self.__fission_source[group]
        return self.__fission_dist.sum()
    
    def __get_dist_error(self,dist,dist_last):
        disp = abs((dist-dist_last)/dist)
        return np.nanmax(disp)

    def __calculate_fission_dist_error(self):
        return self.__get_dist_error(self.__fission_dist,self.__fission_dist_last)

    def __save_result(self,name='result'):
        # Save the flux distribution and the material layout to .vtk
        # Please use ParaView or etc. to visualize it
        self.__get_ordinate()
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
            for group in range(self.__group_num):
                flux = self.__flux[group]
                file_obj.write('SCALARS Flux_group_{} float 1'.format(group)+'\n')
                file_obj.write('LOOKUP_TABLE default'+'\n')
                for i in range(self.__flux_row*self.__flux_col):
                    file_obj.write('{}'.format(flux[i])+'\n')
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

    # def solve_source_iter(self,max_iter,k_tolerance,flux_tolerance,initial_k=1.0):
    #     k = initial_k
    #     self.__define_matrix()
    #     self.__initial_flux()
    #     for i in range(max_iter):
    #         # Update source
    #         if i==0:
    #             self.__define_fission_source()
    #             self.__define_source(k)
    #             last_fission = self.__calculate_fission()
    #         # Solve flux
    #         # Ax = b
    #         for group in range(self.__group_num):
    #             self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
    #         # Update k and source
    #         self.__define_fission_source()
    #         new_fission = self.__calculate_fission()
    #         new_k = k*new_fission/last_fission
    #         # Stopping criterion
    #         if abs(k-new_k)/new_k<=k_tolerance:
    #             print('Iteration {}: Eigenvalue k met the criterion and k = {}'.format(i+1,new_k))
    #             self.__save_result()
    #             break 
    #         else:
    #             print('Iteration {}: Eigenvalue k = {}'.format(i+1,new_k))
    #         if i==max_iter-1:
    #             print('Reached the maximum iteration number{}'.format(i+1))
    #             self.__save_result()
    #         # Update others
    #         last_fission = new_fission
    #         k = new_k
    #         self.__define_source(k)

    def solve_source_iter_correct(self,max_iter,k_tolerance,flux_tolerance,initial_k=1.0):
        # With an acceleration trick
        k = initial_k
        self.__define_matrix()
        self.__initial_flux()
        for i in range(max_iter):
            # Update source
            if i==0:
                self.__define_fission_source()
                last_fission = self.__calculate_fission()
            # Solve flux
            # Ax = b
            for group in range(self.__group_num):
                self.__define_source_group(k,group)
                self.__flux[group] = spsolve(self.__matrix_A[group], self.__source_b[group])
            # Update k and source
            self.__define_fission_source()
            new_fission = self.__calculate_fission()
            new_k = k*new_fission/last_fission
            # Stopping criterion
            fission_dist_error = self.__calculate_fission_dist_error()
            if abs(k-new_k)/new_k<=k_tolerance and fission_dist_error<=flux_tolerance:
                print('Iteration {}: Met the criteria and k = {}, εq = {}'.format(i+1,new_k,fission_dist_error))
                self.__save_result()
                break 
            else:
                print('Iteration {}: Eigenvalue k = {}'.format(i+1,new_k))
                print('Error of fission source εq = {}'.format(fission_dist_error))
            if i==max_iter-1:
                print('Reached the maximum iteration number{}'.format(i+1))
                self.__save_result()
            # Update others
            last_fission = new_fission
            k = new_k
