import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_loss(loss):
  plt.plot(loss,marker='*')
  plt.ylabel('Loss'); plt.xlabel("# of epochs")
  plt.show()

def plot_thetaloss(X,loss,theta,learning_rate):
  for i in range(X.shape[1]):  
    plt.plot(theta[:,i],loss,marker="*")
    plt.title('Learning rate {}'.format(learning_rate))
    plt.ylabel('Loss')
    plt.xlabel('theta {}'.format(i))
    plt.show()

def gradient_descent(X,y,batch_size,optimizer = 'adam',learning_rate=1e-3,beta1=0.1,beta=0.9,n_epochs=200,epsilon = 1e-8):
  bias_col = np.ones((len(X),1))

  try:
    batch_size = int(batch_size)
  except:
    batch_size = len(X)

  try:
    X = np.concatenate((bias_col, X), axis=1)
  except:
    X = np.concatenate((bias_col, X.reshape(-1,1)), axis=1)

  n = float(len(y))

  minima = 1e-3;
  n_feat = X.shape[1]
  thetas = np.array([0]*n_feat); momentum = np.array([0]*n_feat); velocity = np.array([0]*n_feat); sq_grad_avg =  np.array([0]*n_feat)
  theta_updates = np.array([0]*n_feat)
  
  h_values= []
  losses = []
  losses.append(np.linalg.norm(X@thetas-y)**2 / (2*n)) # First error calculation
  for i in range(n_epochs):

    print(y)
    print("****************** Iteration", i," ********************")
    for k in range(0,int(n),batch_size):
      if k+batch_size > n:
        last_elem = int(n)
      else:
        last_elem = k+batch_size
      X_batch, y_batch = X[k:last_elem],y[k:last_elem]

      h = X_batch@thetas
  
      error_vec = (h-y_batch)
      j_theta = np.linalg.norm(error_vec)**2 / (2*batch_size)

      grad_vec  = (error_vec@X_batch)  / (batch_size)

      if optimizer =='adam':
        momentum = (beta1 * momentum) + ((1.0 - beta1) * grad_vec)
        velocity = (beta * velocity) + ((1.0 - beta) * pow(grad_vec,2))

        mhat = momentum / (1.0 - beta1**(i+1))
        vhat = velocity / (1.0 - beta**(i+1))

        update = learning_rate / (pow(vhat,0.5) + epsilon) *mhat
      
      elif optimizer == 'rms_prop':
        sq_grad_sums = np.linalg.norm(grad_vec)
        sq_grad_avg = (sq_grad_avg * beta) + (sq_grad_sums * (1.0-beta))

        alpha = learning_rate / (epsilon + sq_grad_sums)

        update = alpha * grad_vec
      
      elif optimizer == 'ada_grad':
        sq_grad_sums = np.linalg.norm(grad_vec)

        alpha = learning_rate / (epsilon + sq_grad_sums)

        update = alpha * grad_vec

      elif optimizer == 'momentum':
        momentum = (beta * momentum) + (learning_rate * grad_vec)

        update = momentum  

      elif optimizer == 'nag':
        momentum = (beta * momentum) + (learning_rate * grad_vec)

        update = momentum
        thetas = thetas - update

      thetas = thetas - update 
  
    h = X@thetas
    error_vec = h-y
    j_theta = np.linalg.norm(error_vec)**2 / (2*batch_size)
    grad_norm = np.linalg.norm(grad_vec)
  
    print("j = ",j_theta)
    print("Gradient Vector Norm: \n",grad_norm)

    theta_updates = np.vstack([theta_updates, list(thetas)])
  
    j_theta = np.linalg.norm(error_vec)**2 / (2*n)
    losses.append(j_theta)

    if losses[-2] - losses[-1] <= minima:
      print("****************** Training Report ********************")
      print("Result saved at ",i," iterations")

      print("h(x) = y_predict:\n",list(h))
      print("y_actual:\n",list(y))
      h_values.append(h) ## append best values
    
    if j_theta == np.inf:
      print("Model failed to converge, process is terminated")
      return h,losses,theta_updates

  if len(h_values) !=0:
    h = h_values[-1]
  
  print("R2 score of the model :",r2_score(h,y))
  return h,losses,theta_updates