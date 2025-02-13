import numpy as np
import cvxpy as cp

def Setup_Derivative(param):
    ## this function is optional
    return None
   
def Student_Controller_LQR(x_bar, u_bar, x0, Fun_Jac_dt, param):
    

    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = u_bar.shape[0] * u_bar.shape[1]
    n_x = x_bar.shape[0] * x_bar.shape[1]
    n_var = n_u + n_x

    n_eq  = x_bar.shape[1] * u_bar.shape[0] # dynamics
    n_ieq = u_bar.shape[1] * u_bar.shape[0] # input constraints
    
    # define the parameters
    Q = np.eye(4) * 215
    R = np.eye(2) * 49
    Pt = np.eye(4) *220
    # define the cost function
    np.random.seed(1)
    P = np.zeros((n_var, n_var))
    for k in range(u_bar.shape[0]):
        P[k * x_bar.shape[1]:(k+1) * x_bar.shape[1], k * x_bar.shape[1]:(k+1) * x_bar.shape[1]] = Q
        P[n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1], n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1]] = R
    
    P[n_x - x_bar.shape[1]:n_x, n_x - x_bar.shape[1]:n_x] = Pt
    P = (P.T + P) / 2
    q = np.zeros((n_var, 1))
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    for k in range(u_bar.shape[0]):
        
        L_r = param["L_r"]
        L_f = param["L_f"]
        h   = param["h"]
        
        phi = x_bar[k,2]
        v   = x_bar[k,3]
        delta= u_bar[k,0]


        beta = np.arctan((L_r*np.arctan(delta))/(L_r+L_f))
        Ak = np.zeros((4, 4))
        B = np.zeros((4, 2))
        Ak[0, 0] = 1.0
        Ak[0, 1] = 0.0
        Ak[0, 2] = -h*v*np.sin(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        Ak[0, 3] = h*np.cos(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        
        Ak[1, 0] = 0.0
        Ak[1, 1] = 1.0
        Ak[1, 2] = h*v*np.cos(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        Ak[1, 3] = h*np.sin(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        
        Ak[2, 0] = 0.0
        Ak[2, 1] = 0.0
        Ak[2, 2] = 1.0
        Ak[2, 3] = (h * np.arctan(delta)) / (((L_r**2*np.arctan(delta**2))/(L_f+L_r)**2 +1)**0.5*(L_f+L_r))
        
        Ak[3, 0] = 0.0
        Ak[3, 1] = 0.0
        Ak[3, 2] = 0.0
        Ak[3, 3] = 1.0
        
        B[0,0] = 0.0
        B[0,1] = -(h*L_r*v*np.sin(phi+beta))/((delta**2+1)*((L_r**2*np.arctan(delta**2))/(L_r+L_f)**2+1)*(L_r+L_f))
        
        B[1,0] = 0.0
        B[1,1] = (h*L_r*v*np.cos(phi+beta))/((delta**2+1)*((L_r**2*np.arctan(delta**2))/(L_r+L_f)**2+1)*(L_r+L_f))
        
        B[2,0] = 0.0
        B[2,1] = (h*v)/((delta**2+1)*((L_r*22*np.arctan(delta**2))/(L_f+L_r)**2+1)**1.5*(L_f+L_r))
        
        B[3, 1] = 0.0
        B[3, 0] = h
        ## iterate the list of reference trajectory to obtain the linearized dynamics
        # AB = Fun_Jac_dt(x_bar[k, :], u_bar[k, :], param)
        A[k * dim_state:(k+1) * dim_state,      k * dim_state:(k+1) * dim_state]       = Ak # AB[0:dim_state, 0:dim_state]
        A[k * dim_state:(k+1) * dim_state,  (k+1) * dim_state:(k+2) * dim_state]       = -np.eye(dim_state)
        A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]  = B # AB[0:dim_state, dim_state:]
        
       

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)

    cons = [A @ x == b,
            x[0:dim_state] == x0 - x_bar[0, :]
           ]

        
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     cons)
    prob.solve(verbose=False, max_iter = 10000)
    
    return x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]

def Student_Controller_CMPC(x_bar, u_bar, x0, Fun_Jac_dt, param):
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = u_bar.shape[0] * u_bar.shape[1]
    n_x = x_bar.shape[0] * x_bar.shape[1]
    n_var = n_u + n_x

    n_eq  = x_bar.shape[1] * u_bar.shape[0] # dynamics
    n_ieq = u_bar.shape[1] * u_bar.shape[0] # input constraints
    
    # define the parameters
    
    Q = np.eye(4) * 10
    R = np.eye(2) * 1
    Pt = np.eye(4) * 110
    # define the cost function
    np.random.seed(1)
    P = np.zeros((n_var, n_var))
    for k in range(u_bar.shape[0]):
        P[k * x_bar.shape[1]:(k+1) * x_bar.shape[1], k * x_bar.shape[1]:(k+1) * x_bar.shape[1]] = Q
        P[n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1], n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1]] = R
    
    P[n_x - x_bar.shape[1]:n_x, n_x - x_bar.shape[1]:n_x] = Pt
    P = (P.T + P) / 2
    q = np.zeros((n_var, 1))
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    G = np.zeros((n_ieq, n_var))
    ub = np.zeros(n_ieq)
    lb = np.zeros(n_ieq)
    u_lb = np.array([param["a_lim"][0], param["delta_lim"][0]])
    u_ub = np.array([param["a_lim"][1], param["delta_lim"][1]])
    
    for k in range(u_bar.shape[0]):
    
        L_r = param["L_r"]
        L_f = param["L_f"]
        h   = param["h"]
        
        phi = x_bar[k,2]
        v   = x_bar[k,3]
        delta= u_bar[k,0]


        beta = np.arctan((L_r*np.arctan(delta))/(L_r+L_f))
        Ak = np.zeros((4, 4))
        B = np.zeros((4, 2))
        Ak[0, 0] = 1.0
        Ak[0, 1] = 0.0
        Ak[0, 2] = -h*v*np.sin(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        Ak[0, 3] = h*np.cos(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        
        Ak[1, 0] = 0.0
        Ak[1, 1] = 1.0
        Ak[1, 2] = h*v*np.cos(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        Ak[1, 3] = h*np.sin(phi + np.arctan((L_r*np.arctan(delta))/(L_r+L_f)))
        
        Ak[2, 0] = 0.0
        Ak[2, 1] = 0.0
        Ak[2, 2] = 1.0
        Ak[2, 3] = (h * np.arctan(delta)) / (((L_r**2*np.arctan(delta**2))/(L_f+L_r)**2 +1)**0.5*(L_f+L_r))
        
        Ak[3, 0] = 0.0
        Ak[3, 1] = 0.0
        Ak[3, 2] = 0.0
        Ak[3, 3] = 1.0
        
        B[0,0] = 0.0
        B[0,1] = -(h*L_r*v*np.sin(phi+beta))/((delta**2+1)*((L_r**2*np.arctan(delta**2))/(L_r+L_f)**2+1)*(L_r+L_f))
        
        B[1,0] = 0.0
        B[1,1] = (h*L_r*v*np.cos(phi+beta))/((delta**2+1)*((L_r**2*np.arctan(delta**2))/(L_r+L_f)**2+1)*(L_r+L_f))
        
        B[2,0] = 0.0
        B[2,1] = (h*v)/((delta**2+1)*((L_r*22*np.arctan(delta**2))/(L_f+L_r)**2+1)**1.5*(L_f+L_r))
        
        B[3, 1] = 0.0
        B[3, 0] = h
        ## iterate the list of reference trajectory to obtain the linearized dynamics
        # AB = Fun_Jac_dt(x_bar[k, :], u_bar[k, :], param)
        A[k * dim_state:(k+1) * dim_state,      k * dim_state:(k+1) * dim_state]       = Ak # AB[0:dim_state, 0:dim_state]
        A[k * dim_state:(k+1) * dim_state,  (k+1) * dim_state:(k+2) * dim_state]       = -np.eye(dim_state)
        A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]  = B # AB[0:dim_state, dim_state:]
        
        G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]    = np.eye(dim_ctrl)
        ub[k * dim_ctrl:(k+1) * dim_ctrl] = u_ub - u_bar[k, :]
        lb[k * dim_ctrl:(k+1) * dim_ctrl] = u_lb - u_bar[k, :]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)

    cons = [A @ x == b,
            x[0:dim_state] == x0 - x_bar[0, :]
           ]

  
    cons.append(G @ x <= ub)
    cons.append(lb <= G @ x)
        
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     cons)
    prob.solve(verbose=False, max_iter = 10000)
    
    return x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]




