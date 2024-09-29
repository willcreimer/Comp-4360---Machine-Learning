import numpy as np 

def prepend_ones(X):
    n = len(X)
    return np.insert(X, 0, np.ones(n), axis=1)

#error functions
def sse(true, predicted):
    assert(len(true) == len(predicted))

    error = 0
    for i in range(len(true)):
        error = error + ((true[i] - predicted[i]) ** 2)

    return error

def mse(true, predicted):
    return sse(true, predicted) / len(true)

#superclass for linear regressors
class LinearRegressor:
   def __init__(self):
      self.weights = None

   def predict(self, X):
      if self.weights is None:
         print("Model is not trained, cannot predict")
         return None
      
      X_ones = prepend_ones(X)
      return X_ones @ self.weights
   
#closed-form linear regressor implementation
class ClosedLinearRegressor(LinearRegressor):
   def __init__(self):
      super().__init__()

   def train(self, X, t):
      X_ones = prepend_ones(X)
      self.weights = np.linalg.pinv(X_ones) @ t

#SGD linear regressor implementation
class IterativeLinearRegressor(LinearRegressor):
    def __init__(self):
        super().__init__()

    def train(self, X, t, epochs=30, learning_rate = 0.001, initialization='uniform'):
        assert(len(X) == len(t))
        X_ones = prepend_ones(X)
        n, d = X_ones.shape

        #initialize the weights
        if(initialization == 'uniform'):
            self.weights = np.random.uniform(X.min(), X.max(), d)
        elif(initialization == 'normal'):
            self.weights = np.random.normal(size=d)
        elif(initialization == 'zeros'):
            self.weights = np.zeros(d)
        else:
            print("invalid initialization")
            return

        rng = np.random.default_rng()

        errors = np.zeros(epochs)
        for i in range(epochs):
            indices = rng.permutation(n)

            for j in indices:
                e = t[j] - (X_ones[j] @ self.weights)
                self.weights = self.weights + learning_rate * e * X[j]

            error = mse(t, self.predict(X))
            errors[i] = error

        return errors

#guassian rbf    
def rbf(x, centre, width):
    r = np.linalg.norm(x - centre)
    return np.exp(-(r/width)**2)

#non-linear regression implementation
class RBFRegressor:
    def __init__(self, num_centres=5, width=2.5):
        self.lin_reg = ClosedLinearRegressor()
        self.num_centres = num_centres
        self.width = width

    def project(self, X):
        N = X.shape[0]
        Z = np.zeros(shape=(N, self.num_centres))

        for i in range(N):
            for j in range(self.num_centres):
                Z[i][j] = rbf(X[i], self.centres[j], self.width)

        return Z

    def train(self, X, t):
        D = X.shape[1]

        #create centres
        self.centres = np.zeros(shape=(self.num_centres, D))
        for d in range(D):
            xd = X[:,d]
            self.centres[:, d] = np.linspace(xd.min(), xd.max(), self.num_centres)

        #project dataset using the basis functions
        Z = self.project(X)
        
        #train the linear regressor on the projected data
        self.lin_reg.train(Z, t)

    def predict(self, X):
        #project using the basis functions
        Z = self.project(X)

        #put the projected data through the linear regressor
        return self.lin_reg.predict(Z)

#finds the generalization errors for all combinations of parameters (num centres, widths) given   
def evaluate_params(centre_counts, widths, X, t, n_folds=3):
    N = X.shape[0]
    rng = np.random.default_rng()

    errors = np.zeros(shape=(len(centre_counts), len(widths)))

    #try each combination of hyperparameters
    for i in range(len(centre_counts)):
        for j in range(len(widths)):
            centre_count = centre_counts[i]
            width = widths[j]

            reg = RBFRegressor(centre_count, width)

            #split into folds
            shuffled_indices = rng.permutation(N)
            X_shuffled = X[shuffled_indices]
            t_shuffled = t[shuffled_indices]
            X_folds = np.array_split(X_shuffled, n_folds)
            t_folds = np.array_split(t_shuffled, n_folds)

            #use each fold for generalization
            for k in range(len(X_folds)):
                X_train = np.vstack(X_folds[i != k])
                X_test = X_folds[k]

                t_train = np.vstack(t_folds[i!=k])
                t_test = t_folds[k]

                reg.train(X_train, t_train)
                pred = reg.predict(X_test)

                errors[i][j] += sse(t_test, pred)
            
            #take the average over the errors for all folds
            errors[i][j] = errors[i][j] / n_folds

    return errors

def min_indices(matrix):
    return np.unravel_index(np.argmin(matrix), matrix.shape)

#finds the parameters (num centres, widths) with minimum error of all those given
def min_params(centre_counts, widths, X, t, n_folds=3):
    errors = evaluate_params(centre_counts, widths, X, t, n_folds=n_folds)
    indices = min_indices(errors)
    return centre_counts[indices[0]], widths[indices[1]]