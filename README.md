# FDM-SCM-for-neutron-diffusion-alpha-eigenvalue
Test code for 2D transient neutron diffusion equation based on the finite difference method (FDM) and stiffness confinement method (SCM). 

Features
1. Stiffness confinement method (SCM) is applied.
2. Point kinetics equation (PKE) solver is embeded in the code to accelerate the frequency search of SCM
3. Adaptive time stepping SCM is achieved, where PKE solver assists the error and time step control.
