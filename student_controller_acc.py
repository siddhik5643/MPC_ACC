<<<<<<< HEAD
def Student_Controller(t, x, param):
    import numpy as np
    vd = param["vd"]
    v0 = param["v0"]

    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]
    
    ## TODO
    # lam = 
    # alpha =
    # w = 
    # h = 
    # B = 
    # other ...

    ## TODO
    P = np.zeros((2,2))
    
    ## TODO
    A = np.zeros([5, 2])

    ## TODO
    b = np.zeros([5])

    ## TODO
    q = np.zeros([2, 1])
    
    return A, b, P, q
=======
def Student_Controller(t, x, param):
    import numpy as np

    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    lam = 22
    alpha = 0.35
    w = 7050000

    h = 0.5*((x[1]-vd)**2)  
    B   = x[0] - 1.8 * x[1] - 0.5 * (x[1] - v0)**2 / Cdg
    
    P = np.array([[1, 0],
               [0, w]])
    
    q = np.zeros([2, 1])
    

    A = np.array([[(x[1]-vd)/m, -1],
        [(1.8+((x[1]-v0)/(Cdg)))/m, 0],
        [1/m, 0],
        [-1/m, 0],
        [0, -1]])  
    
    b = np.array([-lam*h, alpha*B+(v0-x[1]), Cag, Cdg, 0]) 

    return A, b, P, q
>>>>>>> 453c05b (initial commit)
