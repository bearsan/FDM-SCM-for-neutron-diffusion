import numpy as np
import scipy

'''
The finite difference code for the neutron diffusion equation
Wei Xiao
xiaowei810@foxmail.com
Only for the test of transient algorithms
'''

class diffusion_FDM:
    def __init__(self):
        # self.__mat_col = 1
        # self.__mat_row = 1
        # self.__flux_col = 1
        # self.__flux_row = 1

        self.__boundary = {}
        self.__boundary_beta = {}

        #TODO
        #TEST
        # M1
        XS_scatter = [[0,0.01],[0,0]]
        XS_fission_spec = [1.0,0.0]
        XS_g1 = [1.4,0.01,0.007,XS_scatter[0],XS_fission_spec[0]]
        XS_g2 = [0.4,0.15,0.2,XS_scatter[1],XS_fission_spec[1]]
        XS_M1 = [XS_g1,XS_g2]
        # M2
        XS_scatter = [[0,0.01],[0,0]]
        XS_fission_spec = [1.0,0.0]
        XS_g1 = [1.4,0.01,0.007,XS_scatter[0],XS_fission_spec[0]]
        XS_g2 = [0.4,0.15,0.2,XS_scatter[1],XS_fission_spec[1]]
        XS_M2 = [XS_g1,XS_g2]
        # M3
        XS_scatter = [[0,0.01],[0,0]]
        XS_fission_spec = [1.0,0.0]
        XS_g1 = [1.3,0.008,0.003,XS_scatter[0],XS_fission_spec[0]]
        XS_g2 = [0.5,0.05,0.06,XS_scatter[1],XS_fission_spec[1]]
        XS_M3 = [XS_g1,XS_g2]
        # XS
        self.__XS_ls = [XS_M1,XS_M2,XS_M3]

        self.__group_num = 2 
        self.__mat_col = 2
        self.__mat_row = 2
        self.__flux_row = 3
        self.__flux_col = 3
        
        self.__define_mat_m()
        self.__define_geo_m_x()
        self.__define_geo_m_y()
        self.__initial_flux()


    def __v2m(self,v_i):
        m_j = v_i//self.__flux_row
        m_i = v_i-m_j*self.__flux_row
        return m_i,m_j
    def __m2v(self,m_i,m_j):
        v_i = m_j*self.__flux_row+m_i
        return v_i

    # Material list
    # 1st: material; 2nd: group; 3rd: XS: Dg, Σag, vΣfg, Σs, χ
    def __define_mat_m(self):
        self.mat_m = np.zeros((self.__mat_row,self.__mat_col),dtype=np.int32)
        self.mat_m[0,0] = 0
        self.mat_m[1,0] = 1
        self.mat_m[0,1] = 1
        self.mat_m[1,1] = 2

    #TODO
    def __define_geo_m_x(self):
        # self.geo_m_x = np.ones(self.__mat_row)
        self.geo_m_x = np.array([1,2])
    
    def __define_geo_m_y(self):
        # self.geo_m_y = np.ones(self.__mat_col)
        self.geo_m_y = np.array([2,1])

    def __initial_flux(self):
        # Initial flux storage
        self.__flux = [None]*self.__group_num
        for i in range(self.__group_num):
            self.__flux[i] = np.ones(self.__flux_col*self.__flux_row)

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
        np.savetxt('scattering_term.asu',scatter_term)
    def define_fission_term(self,group_in):
        fission_term = np.zeros(self.__flux_col*self.__flux_row)
        for group_out in range(self.__group_num):
            group_flux = self.__flux[group_out]
            for i in range(self.__flux_row):
                for j in range(self.__flux_col):
                    v_i = self.__m2v(i,j)
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
        np.savetxt('fission_term.asu',fission_term)