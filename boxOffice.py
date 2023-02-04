import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


class boxOffice(object):


    def __init__(self, theta = None, eps = 1e-5, max_iter=10000):
        '''
    
        Initialize our boxOffice object.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        theta : Numpy Array
            Property of self that will hold our value for theta
            which will be used to train most of our models.
        eps : Float
            Property of self that will be used to tell when we
            have reached convergence in our gradient and stochastic
            gradient descent algorithms.
        max_iter : Float
            Property of self that ensures we only loop a specified
            number of iterations in our gradient and stochastic
            gradient descent algorithms.
    
        Returns
        -------
        Nothing
    
        '''
        
        self.theta = theta
        self.eps = eps
        self.max_iter = max_iter


    def fit_NE(self, x, y):
        '''
    
        Run the normal equation on the model.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to train theta.
        y : Numpy Array
            The true labels for the data points that are being trained.
    
        Returns
        -------
        Nothing
    
        '''
        
        x_Tx = np.dot(x.T, x)
        x_Tx_1 = np.linalg.inv(x_Tx)
        x_Ty = np.dot(x.T, y)
        self.theta = x_Tx_1.dot(x_Ty)


    def fit_GD_LR(self, x, y, step_size):
        '''
    
        Run gradient descent with linear regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to train theta.
        y : Numpy Array
            The true labels to have a reference for theta to keep
            updating towards after its prediction.
        step_size : Float
            Controls how quickly we want theta to converge.
    
        Returns
        -------
        Nothing
    
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(2000):
            previous_theta = np.copy(self.theta)
            prediction = np.dot(x, self.theta)
            step = step_size * (x.T.dot(prediction - y))
            self.theta = self.theta - step
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_GD_P(self, x, y, step_size):
        '''
    
        Run gradient descent with poisson regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to train theta.
        y : Numpy Array
            The true labels to have a reference for theta to keep
            updating towards after its prediction.
        step_size : Float
            Controls how quickly we want theta to converge.
    
        Returns
        -------
        Nothing
    
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(2000):
            previous_theta = np.copy(self.theta)
            prediction = np.exp(np.dot(x, self.theta))
            step = step_size * (x.T.dot(prediction - y))
            self.theta = self.theta - step
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_SGD_LR(self, x, y, step_size):
        '''
    
        Run stochastic gradient descent with linear regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to train theta.
        y : Numpy Array
            The true labels to have a reference for theta to keep
            updating towards after its prediction.
        step_size : Float
            Controls how quickly we want theta to converge.
    
        Returns
        -------
        Nothing
    
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(1000):
            previous_theta = np.copy(self.theta)
            for j in range(d):
                prediction = np.dot(x[j], self.theta)
                step = step_size * (x[j] * (prediction - y[j]))
                self.theta = self.theta - np.reshape(step, (len(step),1))
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_SGD_P(self, x, y, step_size):
        '''
    
        Run stochastic gradient descent with poisson regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to train theta.
        y : Numpy Array
            The true labels to have a reference for theta to keep
            updating towards after its prediction.
        step_size : Float
            Controls how quickly we want theta to converge.
    
        Returns
        -------
        Nothing
    
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(1000):
            previous_theta = np.copy(self.theta)
            for j in range(d):
                prediction = np.exp(np.dot(x[j], self.theta))
                step = step_size * (x[j] * (prediction - y[j]))
                self.theta = self.theta - np.reshape(step, (len(step),1))
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def predict_LR(self, x):
        '''
    
        Gets a prediction using Linear Regression.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to predict a set
            of y-values.
        
        Returns
        -------
        y : Numpy Array
            The predicted values using the inputs.
    
        '''
        
        y = x.dot(self.theta)
        
        y[y < 0] = 0
        
        return y


    def predict_P(self, x):
        '''
    
        Gets a prediction using Poisson.
    
        Parameters
        ----------
        self : boxOffice(object)
            Holds the boxOffice object.
        x : Numpy Array
            Holds the data points that will be used to predict a set
            of y-values.
    
        Returns
        -------
        y : Numpy Array
            The predicted values using the inputs.
    
        '''
    
        y = np.exp(x.dot(self.theta))
        
        y[y < 0] = 0
        
        return y
    

def rf_regressor(x_train, y_train, x_valid, y_valid):
    '''
    
    Gets a prediction using the random forest regressor.

    Parameters
    ----------
    x_train : Numpy Array
        Holds the data points to train.
    y_train : Numpy Array
        Holds the y-labels to train with x_train.
    x_valid : Numpy Array
        Holds the data points that will be used in the built-in
        predict function.
    y_valid : Numpy Array
        Holds the true y-values for the valid data points, which we
        will compare with our prediction.

    Returns
    -------
    y_pred : Numpy Array
        The y-values attained from the built-in predict function.

    '''
    
    fit_model = '\nRANDOM FOREST REGRESSOR'
    
    rfr_model = RandomForestRegressor()
    
    rfr_model.fit(x_train, y_train)
    y_pred = rfr_model.predict(x_valid)
    
    accuracy = score(y_pred, y_valid)
    
    print(fit_model)
    print(accuracy)
    print()
    
    return y_pred


def normal_equation(x_train, y_train, x_valid, y_valid, model):
    '''
    
    Gets a prediction using the normal equation.

    Parameters
    ----------
    x_train : Numpy Array
        Holds the data points to train.
    y_train : Numpy Array
        Holds the y-labels to train with x_train.
    x_valid : Numpy Array
        Holds the data points that will be used in our predict function.
    y_valid : Numpy Array
        Holds the true y-values for the valid data points, which we
        will compare with our prediction.
    model : boxOffice(object)
        Holds the boxOffice object.

    Returns
    -------
    y_pred : Numpy Array
        The y-values attained from our predict function.

    '''
    
    fit_model = 'NORMAL EQUATION'
    
    model.__init__()
    
    model.fit_NE(x_train, y_train)
    y_pred = model.predict_LR(x_valid)
    
    accuracy = score(y_pred, y_valid)
    
    print(fit_model)
    print(accuracy)
    print()
    
    return y_pred


def gradient_descent(x_train, y_train, x_valid, y_valid, model, model_type):
    '''
    
    Gets a prediction using gradient descent.
    The commented out section was used to find the ideal step_size each
    time we added, removed, or modified a feature.

    Parameters
    ----------
    x_train : Numpy Array
        Holds the data points to train.
    y_train : Numpy Array
        Holds the y-labels to train with x_train.
    x_valid : Numpy Array
        Holds the data points that will be used in our predict function.
    y_valid : Numpy Array
        Holds the true y-values for the valid data points, which we
        will compare with our prediction.
    model : boxOffice(object)
        Holds the boxOffice object.
    model_type : String
        Holds either 'lr' for Linear Regression, or 'p' for Poisson.

    Returns
    -------
    y_pred : Numpy Array
        The y-values attained from our predict function.

    '''
    
    if (model_type == 'lr'):
        
        fit_model = 'GRADIENT DESCENT - LINEAR REGRESSION'
        
        '''
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 5:
            if (ideal < 10):
                increment = 0.1
            elif (ideal < 100):
                increment = 1
            elif (ideal < 1000):
                increment = 10
            elif (ideal < 10000):
                increment = 100
            elif (ideal < 100000):
                increment = 1000
            elif (ideal < 1000000):
                increment = 10000
            elif (ideal < 10000000):
                increment = 100000
            elif (ideal < 100000000):
                increment = 1000000
            elif (ideal < 1000000000):
                increment = 10000000
                
            model.__init__()
            step_size = ideal * (10 ** -15)
            model.fit_GD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            y_pred_lr = model.predict_LR(x_valid)
            
            temp = score(y_pred_lr, np.reshape(y_valid.values, (len(y_valid),1)))
        
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
            else:
                max_reached += 1
            ideal += increment
        print(max_score)
        '''
        
        step_size = 1.9e-10
        
        model.__init__()
        
        model.fit_GD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
        y_pred = model.predict_LR(x_valid)
        
        accuracy = score(y_pred, np.reshape(y_valid.values, (len(y_valid),1)))
        
        print(fit_model)
        print(accuracy)
        print()
    
    elif (model_type == 'p'):
    
        fit_model = 'GRADIENT DESCENT - POISSON'
        
        '''
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 3:
            if (ideal < 10):
                increment = 0.1
            elif (ideal < 100):
                increment = 1
            elif (ideal < 1000):
                increment = 10
            elif (ideal < 10000):
                increment = 100
            elif (ideal < 100000):
                increment = 1000
            elif (ideal < 1000000):
                increment = 10000
            elif (ideal < 10000000):
                increment = 100000
            elif (ideal < 100000000):
                increment = 1000000
            elif (ideal < 1000000000):
                increment = 10000000
                
            model.__init__()
            step_size = ideal * (10 ** -15)
            model.fit_GD_P(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            y_pred_lr = model.predict_P(x_valid)
            
            temp = score(y_pred_lr, np.reshape(y_valid.values, (len(y_valid),1)))
        
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
            else:
                max_reached += 1
            ideal += increment
        print(max_score)
        '''
            
        step_size = 9.4e-12
        
        model.__init__()
        
        model.fit_GD_P(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
        y_pred = model.predict_P(x_valid)
        
        accuracy = score(y_pred, np.reshape(y_valid.values, (len(y_valid),1)))
        
        print(fit_model)
        print(accuracy)
        print()
        
    return y_pred


def stochastic_gradient_descent(x_train, y_train, x_valid, y_valid, model, model_type):
    '''
    
    Gets a prediction using stochastic gradient descent.
    The commented out section was used to find the ideal step_size each
    time we added, removed, or modified a feature.

    Parameters
    ----------
    x_train : Numpy Array
        Holds the data points to train.
    y_train : Numpy Array
        Holds the y-labels to train with x_train.
    x_valid : Numpy Array
        Holds the data points that will be used in our predict function.
    y_valid : Numpy Array
        Holds the true y-values for the valid data points, which we
        will compare with our prediction.
    model : boxOffice(object)
        Holds the boxOffice object.
    model_type : String
        Holds either 'lr' for Linear Regression, or 'p' for Poisson.

    Returns
    -------
    y_pred : Numpy Array
        The y-values attained from our predict function.

    '''
    
    if (model_type == 'lr'):
        
        fit_model = 'STOCHASTIC GRADIENT DESCENT - LINEAR REGRESSION'
        
        '''
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 1
        max_reached = 0
        
        while max_reached < 1:
            if (ideal < 10):
                increment = 1
            elif (ideal < 100):
                increment = 10
            elif (ideal < 1000):
                increment = 100
            elif (ideal < 10000):
                increment = 1000
            elif (ideal < 100000):
                increment = 10000
            elif (ideal < 1000000):
                increment = 100000
            elif (ideal < 10000000):
                increment = 1000000
            elif (ideal < 100000000):
                increment = 10000000
            elif (ideal < 1000000000):
                increment = 100000000
            
            model.__init__()
            step_size = ideal * (10 ** -15)
            model.fit_SGD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            y_pred_lr = model.predict_LR(x_valid)
            
            temp = score(y_pred_lr, np.reshape(y_valid.values, (len(y_valid),1)))
        
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
            else:
                max_reached += 1
            ideal += increment
        print(max_score)
        '''
        
        step_size = 2e-10
        
        model.__init__()
        
        model.fit_SGD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
        y_pred = model.predict_LR(x_valid)
        
        accuracy = score(y_pred, np.reshape(y_valid.values, (len(y_valid),1)))
        
        print(fit_model)
        print(accuracy)
        print()
    
    elif (model_type == 'p'):
        
        fit_model = 'STOCHASTIC GRADIENT DESCENT - POISSON'
        
        '''
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 1
        max_reached = 0
        
        while max_reached < 2:
            if (ideal < 10):
                increment = 1
            elif (ideal < 100):
                increment = 10
            elif (ideal < 1000):
                increment = 100
            elif (ideal < 10000):
                increment = 1000
            elif (ideal < 100000):
                increment = 10000
            elif (ideal < 1000000):
                increment = 100000
            elif (ideal < 10000000):
                increment = 1000000
            elif (ideal < 100000000):
                increment = 10000000
            elif (ideal < 1000000000):
                increment = 100000000
            
            model.__init__()
            step_size = ideal * (10 ** -15)
            model.fit_SGD_P(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            y_pred_lr = model.predict_P(x_valid)
            
            temp = score(y_pred_lr, np.reshape(y_valid.values, (len(y_valid),1)))
        
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
            else:
                max_reached += 1
            ideal += increment
        print(max_score)
        '''
        
        step_size = 4e-11
        
        model.__init__()
        
        model.fit_SGD_P(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
        y_pred = model.predict_P(x_valid)
        
        accuracy = score(y_pred, np.reshape(y_valid.values, (len(y_valid),1)))
        
        print(fit_model)
        print(accuracy)
        print()
    
    return y_pred

    
def score(y_pred, y_true):
    '''
    
    Creates and labels the plot given the input'

    Parameters
    ----------
    y_true : Numpy Array
        Holds the true values for the box office results.
    y_pred : Numpy Array
        Holds the predicted values for the box office results.

    Returns
    -------
    score : Float
        Using the least squares method, attain a value for accuracy

    '''
    
    u = np.average(np.square(y_true - y_pred))
    v = np.average(np.square(y_true - y_true.mean()))
    
    score =  1 - (u/v)
    
    return score


def make_plot(y_true, y_pred, title):
    '''
    
    Creates and labels the plot given the input'

    Parameters
    ----------
    y_true : Numpy Array
        Holds the true values for the box office results.
    y_pred : Numpy Array
        Holds the predicted values for the box office results.
    title: String
        The title of the diagram.

    Returns
    -------
    Nothing

    '''
    
    plt.scatter(y_true, y_pred, alpha=0.4, c='blue', label=title)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()
    plt.clf()


def add_intercept(x):
    '''
    
    Add a column of 1's to the input'

    Parameters
    ----------
    x : Numpy Array
        Holds all the data except the y label.

    Returns
    -------
    y : Numpy Array
        Copy of x with the addition of a column of 1's.

    '''
    
    y = np.zeros((x.shape[0], x.shape[1] + 1))
    y[:, 0] = 1
    y[:, 1:] = x

    return y


def main(data_path):
    
    # These are variables to store the features that will be dropped for
    # each model. All of them contain 'yb', as this is our y label.
    
    columns_rf = ['yb', 'xl', 'ah', 'xd', 'ar']
    columns_ne = ['yb', 'ar', 'xla2', 'aa', 'xd', 'br', 'xla1', 'xl', 'ba']
    columns_gd_lr = ['yb', 'xl', 'ar', 'xla2']
    columns_gd_p = ['yb', 'xla2', 'xd']
    columns_sgd_lr = ['yb', 'xl', 'ar', 'xla2']
    columns_sgd_p = ['yb']

    columns = [columns_rf, columns_ne, columns_gd_lr,
               columns_gd_p, columns_sgd_lr, columns_sgd_p]

    #for i in range (len(columns)):
    
    for k in range (1):
        
        i = 3
        
        data = pd.read_csv(data_path)
        
        if (i == 0):
            
            data['xm_sin'] = np.sin(data['xm'])
            
        elif (i == 1):
            
            data['xb_squared'] = data['xb']**2
            data['xd_squared'] = data['xd']**2
            
        elif (i == 3):
            
            data['ah_squared'] = data['ah']**2
            
        elif (i == 5):
            
            data['xla1_squared'] = data['xla1']**2
        
        y = data['yb']
        x = data.drop(columns[i], axis = 1)
        
        x_train, x_valid_test, y_train, y_valid_test = train_test_split(x, y, test_size = 0.30, random_state = 101)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size = 0.33, random_state = 101)
        
        if (i == 0):
            
            rf_pred = rf_regressor(x_train, y_train, x_test, y_test)
            make_plot(y_test, rf_pred, 'Random Forest')

        else:
            
            model = boxOffice()
            x_train = add_intercept(x_train)
            x_valid = add_intercept(x_valid)
            x_test = add_intercept(x_test)
            
            if (i == 1):
                
                normal_pred = normal_equation(x_train, y_train, x_test, y_test, model)
                make_plot(y_test, normal_pred, 'Normal Equation')

            if (i == 2):
                
                gd_lr_pred = gradient_descent(x_train, y_train, x_test, y_test, model, 'lr')
                make_plot(y_test, gd_lr_pred, 'GD - Linear Regression')

            if (i == 3):
                
                gd_p_pred = gradient_descent(x_train, y_train, x_test, y_test, model, 'p')
                make_plot(y_test, gd_p_pred, 'GD - Poisson')

            if (i == 4):
                
                sgd_lr_pred = stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, 'lr')
                make_plot(y_test, sgd_lr_pred, 'SGD - Linear Regression')

            if (i == 5):
                
                sgd_p_pred = stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, 'p')
                make_plot(y_test, sgd_p_pred, 'SGD - Poisson')


if __name__ == '__main__':
    main(data_path='data.csv')