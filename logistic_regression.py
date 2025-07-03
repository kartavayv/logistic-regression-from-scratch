import matplotlib.pyplot as plt
import numpy as np
class logistic_regression():
  def __init__(self,n_iter=100,l_rate=0.001, penalty=None, C=0.0001, use_polynomial=False, degree=1, normalize=False):
    self.n_iter = n_iter
    self.l_rate = l_rate
    self.costs = []      #i like doing this
    self.params = {}
    self.grads = {}
    self.penalty = penalty
    self.reg_strength = (1/C) # it's 1/(reg strength)
    self.use_polynomial = use_polynomial
    self.degree = degree
    self.normalize = normalize

  #BEGINNING STUFF
  def dimensions(self,X,Y):
    n_x = X.shape[1]
    n_y = Y.shape[1]

    return (n_x,n_y)

  def ini_params(self,n_x,n_y):
    W = np.random.randn(n_x,n_y)*0.01 # it's going to be XW instead of WX
    b = 0 #brodcasting will take care of itself!

    params = {
        "W": W,
        "b": b
    }

    self.params = params

  def sigmoid(self,calcs):
    return 1 / (1+np.exp(-calcs))

  ###COST CALCULATIONS
  def cost_func(self,y_pred,Y):
    p1 = np.multiply(Y,np.log(y_pred))
    p2 = np.multiply((1-Y),np.log(1-y_pred))

    # cost = np.sum(-p1 - p2) #summing up errors on each example is necessary!!!
    m = Y.shape[0]

    if self.penalty == "l1":
      cost = (1/m) * (np.sum(-p1 - p2) + self.reg_strength * (self.reg_l1()))
    elif self.penalty == "l2":
      cost = (1/m) * (np.sum(-p1 - p2) + (self.reg_strength/2) * (self.reg_l2()))
    else:
      cost = (1/m) * (np.sum(-p1 - p2))

    return cost
  
  ###REGULARIZERS  
  def reg_l1(self):
    W = self.params['W']
    return np.sum(np.abs(W))

  def reg_l2(self):
    W = self.params['W']
    return np.sum(np.square(W))
  
  ###GRADIENT DESCENT STUFF

  def grad_des(self,X, Y, y_pred):
    m = Y.shape[0]

    if self.penalty == "l1":
      dW = (1/m) * (np.dot(X.T, (y_pred-Y)) + self.reg_strength * self.params["W"])
    elif self.penalty == "l2":
      dW = (1/m) * (np.dot(X.T, (y_pred-Y)) + self.reg_strength * np.sign(self.params["W"]))
    else:
      dW = (1/m) * (np.dot(X.T, (y_pred-Y)))

    db = (1/m) * np.sum((y_pred-Y))

    grads = {
        "dW": dW,
        "db": db
    }

    self.grads = grads

  
  ###FORWARD PROP AND STUFF
  
  def forward_prop(self,X,Y):
    W = self.params["W"]
    b = self.params["b"]

    res = np.dot(X,W) + b

    y_pred = self.sigmoid(res)
    return y_pred

  def fit(self,X,Y):
    n_iter = self.n_iter
    l_rate = self.l_rate
    
    if self.use_polynomial:
      X = self._polynomial_features(X)
    if self.normalize:
      X = self.normal(X)

    n_x,n_y = self.dimensions(X,Y)
    self.ini_params(n_x,n_y)
    for i in range(n_iter):
      y_pred = self.forward_prop(X,Y)
      cost = self.cost_func(y_pred,Y)
      self.costs.append(cost)
      self.grad_des(X,Y,y_pred)

      dW = self.grads['dW']
      db = self.grads['db']

      self.params["W"] = self.params['W'] - (self.l_rate*dW) #thought of making code more readable
      self.params["b"] = self.params['b'] - (self.l_rate*db)


      if i%100 == 0:
        print(f"Cost of the iteration {i} is {cost}")

    final_vals = {
        "W": self.params['W'],
        "b": self.params["b"]
    }

    return self.costs, final_vals

  def predict(self,X):
    model_params = self.params

    if self.use_polynomial:
      X = self._polynomial_features(X)
    if self.normalize:
      X = self.normal(X)

    W = model_params['W']
    b = model_params['b']

    calcs = np.dot(X,W) + b
    preds = (self.sigmoid(calcs) > 0.5).astype(int)

    return preds
  
  ### MODEL EVALUATORS

  def predict_proba(self,X):
    model_params = self.params
    W = model_params['W']
    b = model_params['b']
    calcs = np.dot(X,W) + b

    preds = self.sigmoid(calcs)
    probs = np.column_stack((1-preds,preds)) # a good use of column stack!!! and stacking itself, using arrays ! arrays inside lists

    return probs

  def score(self,X,Y):

    preds = self.predict(X)
    right_preds = (preds==Y).astype(int)

    score = len(right_preds) / len(Y)

    return score

  def precision(self, Y, y_pred):
    TP = np.sum((Y==1)&(y_pred==1))
    FP = np.sum((Y==0)&(y_pred==1))

    return (TP / (TP+FP))

  def recall(self,Y,y_pred):
    TP = np.sum((Y==1)&(y_pred==1))
    FN = np.sum((Y==1)&(y_pred==0))

    return (TP/(TP+FN))

  def f1(self,Y,y_pred):
    P = self.precision(Y,y_pred)
    R =  self.recall(Y,y_pred)

    F1 = (2*P*R) / (P+R)

    return F1

  def confusion_mat(self,Y,y_pred):
    # return np.array([[np.sum(np.intersect1d(Y,y_pred)), np.sum(y_pred==1)-np.sum(np.intersect1d(Y,y_pred))],[np.sum(Y==1)-np.sum(y_pred==1),np.sum(np.abs(np.intersect1d(Y,y_pred)-1))]]) #this was a tough one!!!, doesn't work!!!
    TP = np.sum((Y==1) & (y_pred==1))
    TN = np.sum((Y==0) & (y_pred==0))
    FP = np.sum((Y==0) & (y_pred==1))
    FN = np.sum((Y==1) & (y_pred==0))

    return np.array([[TP,FN],[FP,TN]])
    
  
  def plot_learning_curve(self):
    plt.plot(np.arange(self.n_iter),self.costs)
    plt.title("COST vs EPOCH") #epoch =  one pass through dataset
    plt.xlabel("EPOCHS")
    plt.ylabel("COST")
    plt.show()
  
  ###FEATURE ENGINEERING AND NORMALIZATION

  def _polynomial_features(self, X):
    degree = self.degree
    X_copy = X.copy()

    for i in range(2,degree+1):
      col = np.power(X_copy,i)
      X = np.column_stack((X,col))
    
    return X
  
  def normal(self,X):
    for i in range(len(X[0])):
      X_min = np.min(X[i])
      X_std = np.std(X[i])
      X[i] = (X[i] - X_min) / X_std

    return X