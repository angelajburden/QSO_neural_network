import numpy as np
import scipy.optimize

#NB this code is based around the matlab code of Andrew Ng his machine learning course.
Nfeval = 1

def cost(theta,*args):
    return CostFunction(theta, *args)[0]
 
def grad(theta,*args):
    return CostFunction(theta, *args)[1]
          
def trainReg(X, y,input_layer_size, hidden_layer_size,num_labels, lamparam, it_no):
    print('\nInitializing Neural Network Parameters ...\n')
    print('reloaded\n')
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size)
    initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels)
    #Unroll parameters
    tTheta1 = initial_Theta1.flatten()
    tTheta2 = initial_Theta2.flatten()
    tTheta3 = initial_Theta3.flatten()
    tTheta1 = np.reshape(tTheta1,(tTheta1.shape[0],1))
    tTheta2 = np.reshape(tTheta2,(tTheta2.shape[0],1))
    tTheta3 = np.reshape(tTheta3,(tTheta3.shape[0],1))
    initial_nn_params = np.vstack((tTheta1,tTheta2,tTheta3))
    print('\nTraining Neural Network... \n')
    args = (input_layer_size,hidden_layer_size,num_labels, X, y, lamparam)
    print("define args\n")
    nn_params2 = scipy.optimize.fmin_cg(cost, x0=initial_nn_params, args=args, fprime=grad, maxiter=it_no, full_output=1)
    nn_params2 = np.array(nn_params2)
    nn_params = nn_params2[0]
    print("returning optimised parameters\n")
    return nn_params
    

def CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamparam):
    v1 =(hidden_layer_size * (input_layer_size + 1))
    v2 =(hidden_layer_size * (hidden_layer_size + 1))
    Theta1 = np.reshape(nn_params[0:v1], (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[v1:(v1+v2)], (hidden_layer_size, (hidden_layer_size + 1)))
    Theta3 = np.reshape(nn_params[(v1+v2):],(num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    J = np.array([0.0])
    Theta1_grad = np.zeros((Theta1.shape))
    Theta2_grad = np.zeros((Theta2.shape))
    Theta3_grad = np.zeros((Theta3.shape))
    X1s = np.ones((m,1))
    X = np.append(X1s,X, 1)
    a1 = np.zeros((1,input_layer_size))
    a2 = np.zeros((m,hidden_layer_size+1))   
    a3 = np.zeros((m,hidden_layer_size+1))
    h_theta = np.zeros((m,num_labels))
    
    d2 = np.zeros((m,hidden_layer_size))
    d3 = np.zeros((m,hidden_layer_size))
    d4 = np.zeros((m,num_labels))
    minval=0.0000000001 #just incase a val v.close zero returned (can't take log)
    for i in range(0,m):
        a1 = X[i,:]
        z2 = np.dot(a1, Theta1.transpose())
        a2[i,0]=1
        a2[i,1:] = sigmoid(z2)
        z3 = np.dot(a2[i,:],Theta2.transpose())
        a3[i,0]=1
        a3[i,1:] = sigmoid(z3)
        z4 = np.dot(a3[i,:],Theta3.transpose())
        h_theta[i] = sigmoid(z4) 
        
        #compute the cost function
        for j in range(0,num_labels):
            
            if h_theta[i,j] < minval:
                h_theta[i,j] = minval
            if h_theta[i,j] > 1.-minval:
                h_theta[i,j] = 1.-minval    
            if num_labels ==1:
                J = J - y[i]*np.log(h_theta[i,j]) - (1-y[i])*np.log(1-h_theta[i,j])
            if num_labels >1:
                J = J - y[i,j]*np.log(h_theta[i,j]) - (1-y[i,j])*np.log(1-h_theta[i,j])            
    #compute the regularisation          
    C1 = np.multiply(Theta1[:,1:],Theta1[:,1:])
    C1sum = np.sum(C1)
    C2 = np.multiply(Theta2[:,1:],Theta2[:,1:])
    C2sum = np.sum(C2)
    C3 = np.multiply(Theta3[:,1:],Theta3[:,1:])
    C3sum = np.sum(C3)
    J = J/m + (lamparam/(2.*m))*(C1sum + C2sum + C3sum)
    
    #compute the gradient
    for i in range(0,m):        
        d4[i] = h_theta[i] -y[i]        
        z3 = np.dot(a2[i,:],Theta2.transpose())       
        temp = np.dot(d4[i,:],Theta3[:,1:])        
        d3[i,:] = np.multiply(temp,sigmoidGradient(z3)) 
        z2 = np.dot(X[i,:],Theta1.transpose())
        temp = np.dot(d3[i,:],Theta2[:,1:])     
        d2[i,:] =  np.multiply(temp,sigmoidGradient(z2))
        
        tri1 = np.dot(X[i,:,None],d2[i,:,None].T)
        tri2 = np.dot(a2[i,:,None],d3[i,:,None].T)
        tri3 = np.dot(a3[i,:,None],d4[i,:,None].T)
        
        Theta1_grad = Theta1_grad + (tri1.T)/m
        Theta2_grad = Theta2_grad + (tri2.T)/m
        Theta3_grad = Theta3_grad + (tri3.T)/m
  
    t1 = Theta1[:,1:]
    greg1 = (lamparam/m)* t1
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + greg1 

    t2 = Theta2[:,1:]
    greg2 = (lamparam/m)* t2
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + greg2 
    
    t3 = Theta3[:,1:]
    greg3 = (lamparam/m)* t3
    Theta3_grad[:,1:] = Theta3_grad[:,1:] + greg3

    tTheta1_g = Theta1_grad.flatten()
    tTheta2_g = Theta2_grad.flatten()
    tTheta3_g = Theta3_grad.flatten()
    
    tTheta1_g = np.reshape(tTheta1_g,(tTheta1_g.shape[0],1))
    tTheta2_g = np.reshape(tTheta2_g,(tTheta2_g.shape[0],1))
    tTheta3_g = np.reshape(tTheta3_g,(tTheta3_g.shape[0],1))
    
    grad = np.vstack((tTheta1_g,tTheta2_g,tTheta3_g))
    grad = np.ndarray.flatten(grad)

    return J, grad

def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    INIT_EPSILON = 0.1
    W = np.random.random((L_out,1 + L_in)) * (2 * INIT_EPSILON) - INIT_EPSILON
    return W

def predict(Theta1, Theta2, Theta3, X):
    m = X.shape[0]
    input_layer_size = X.shape[1]
    p = np.zeros((m, 1))
    X1s = np.ones((m,1))
    X = np.append(X1s,X, 1)   
    h1 = sigmoid(np.dot(X,Theta1.transpose()))    
    h1 = np.append(X1s,h1, 1)     
    h2 = sigmoid(np.dot(h1,Theta2.transpose())) 
    h2 = np.append(X1s,h2, 1)     
    h3 = sigmoid(np.dot(h2, Theta3.transpose()))    
    return h3

def sigmoid(z):
    g =  1./(1 + np.exp (-z))
    return g  

def sigmoidGradient(z):    
    x = 1./(1 + np.exp(-z))
    return np.multiply(x, (1.-x))
    
def validationCurve(X, y, Xval, yval, input_layer_size, hidden_layer_size,num_labels, it_no):
    lambda_vec = np.array(([0.02]))#[0.008], [0.01], [0.02], [0.03],
    error_train = np.zeros((lambda_vec.shape[0], 1))
    error_val = np.zeros((lambda_vec.shape[0], 1))
    for i in range(0,lambda_vec.shape[0]):
        lamparam = lambda_vec[i]
        print('lambda = ', lamparam)
        nn_params = trainReg(X, y,input_layer_size, hidden_layer_size, num_labels, lamparam, it_no)
        [error_train[i], g] = CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
        [error_val[i], g] = CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xval, yval, 0)
    print( np.c_[lambda_vec, error_train, error_val])
    np.savetxt('Validation_lambda_file_200it_2.txt', np.c_[lambda_vec, error_train, error_val],fmt='%1.3f')
    return lambda_vec, error_train, error_val
    
def computeNumericalGradient(J, theta):                
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in xrange(theta.size):        
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

    
def checkNNGradients(lamparam):
#check gradients of a small sample analytically
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    #generate 'random' test data    
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(hidden_layer_size, hidden_layer_size)
    Theta3 = debugInitializeWeights(num_labels, hidden_layer_size)
    
    # generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y =  np.zeros((m,num_labels))
    for ival in range(0,num_labels):
        y[:,ival]  = 1 + np.mod(range(m), num_labels)
        
    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'),\
                Theta2.reshape(Theta2.size, order='F'), Theta3.reshape(Theta3.size, order='F')))
                
    # Short hand for cost function
    def costFunc(p):
        return CostFunction(p, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lamparam)

    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    for i in range(0, grad.shape[0]):
        print((i, numgrad[i], grad[i]))
        
    print(['The above two columns you get should be very similar.\n' 
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n'])
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print(['If backpropagation correct, then \n'
         'relative difference will be less than 1e-9). \n'
         '\nRelative Difference: %g\n'], diff)

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    numW=fan_out*(1 + fan_in)
    W = np.reshape(np.sin(range(1,numW+1)), W.shape) / 10
    return W
    
def learningCurve(X, y, Xval, yval,input_size,hidden_size,num_labels,lamparam, it_no):
    m = X.shape[0]
    no_data = 10
    error_train = np.zeros((no_data, 1))
    error_val   = np.zeros((no_data, 1))
    j_val   = np.zeros((no_data, 1))
    for i in range(no_data):
        j_val[i]= (i+1)*m/float(no_data)
        theta_train = trainReg(X[0:j_val[i],:], y[0:j_val[i],:],input_size,hidden_size,num_labels,lamparam, it_no)
        [error_train[i], gradx] = CostFunction(theta_train, input_size, hidden_size, num_labels, X[0:j_val[i],:], y[0:j_val[i],:], 0)
        [error_val[i], gradx] = CostFunction(theta_train, input_size, hidden_size, num_labels, Xval, yval, 0)  
    np.savetxt('Learning_amountdata_file_100it_0.5.txt', np.c_[j_val, error_train, error_val],fmt='%1.3f') 
    return error_val, error_train
##############################################