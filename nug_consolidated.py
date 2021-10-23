# Import required libraries and packages
from numpy.typing import _128Bit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
import sqlite3 as sql
from patsy import cr

def sigmoid(x, a):
    return  a + 1/(0.033+x**2)

def linear(x, coef, intercept):
    return coef * x + intercept

def cubic(x, a, b, c, d):
    return a + b * x + c * np.square(x) + d * np.power(x,3)

def quadratic(x, a):
    """Fitting function to return the calculation of quadratic model.

    Parameters
    ----------
    x : array
        Test data to be calculated as X data point.
    a : float
        Additional parameter to be used as multiplier in the formula.

    Returns
    -------
    float
        Calculated value from the quadratic formula.
    """
    return a * np.square(x-10)

def predict(func, x_test, popt):
    """Function to predict the built model using test data.

    Parameters
    ----------
    func : string
        Model formula, possible values are quadratic, linear, cubic, and sigmoid.
    x_test : array
        Test data to be calculated as X data point.
    popt : array
        Optimized parameters.

    Returns
    -------
    y_hat : float
        Calculated value from the prediction formula.

    """
    y_hat = 0
    if func == "quadratic":
        y_hat = quadratic(x_test, popt[0])
    elif func == "linear":
        y_hat = linear(x_test, popt[0], popt[1])
    elif func == "cubic":
        y_hat = cubic(x_test, popt[0], popt[1], popt[2], popt[3])
    elif func == "sigmoid":
        y_hat = sigmoid(x_test, popt[0])
    else:
        y_hat = 1

    return y_hat

# Function to evaluate the model
def evaluation(y_hat, y_test):
    MSE = np.mean((y_hat - y_test) ** 2)
    RMSE = np.sqrt(MSE)
    return(RMSE)

# Function to plot the data
def plotting(x_train_data, y_train_data, y_hat):
    plt.plot(x_train_data, y_train_data, "ro")
    plt.plot(x_train_data, y_hat, color="blue")
    plt.show()

# Function to return key for any value in dictionary
def get_key(val, dict):
    for key, value in dict.items():
         if val == value[0]:
             return key
 
    return "Key doesn't exist"

# Function to return smallest record in dictionary
def get_smallest_RMSE(dict):
    min_RMSE_list = list(min(dict.values()))
    min_RMSE_val = min_RMSE_list[0]
    min_RMSE_x = min_RMSE_list[1]
    min_RMSE_y = min_RMSE_list[2]
    min_RMSE_delta = min_RMSE_list[3]
    min_RMSE_key = get_key(min_RMSE_val, dict)

    return {'x': min_RMSE_x, 'y': min_RMSE_y, 'delta_y': min_RMSE_delta, 'ideal_function': min_RMSE_key}

# Function to return smallest record in dictionary (test)
def get_smallest_RMSE2(dict):
    min_RMSE_val = list(min(dict.values()))[0]
    min_RMSE_x = list(min(dict.values()))[1]
    min_RMSE_y = list(min(dict.values()))[2]
    min_RMSE_y_hat = list(min(dict.values()))[3]
    min_RMSE_delta = list(min(dict.values()))[4]
    min_RMSE_key = get_key(min_RMSE_val, dict)

    return {'x': min_RMSE_x, 'y': min_RMSE_y, 'y_hat': min_RMSE_y_hat, 'delta_y': min_RMSE_delta, 'ideal_function': min_RMSE_key}

def main():
    # Read the csv files and load it to dataframes
    train_df = pd.read_csv(os.path.join(os.getcwd(),"data\\train.csv"))
    ideal_df = pd.read_csv(os.path.join(os.getcwd(),"data\ideal.csv"))
    test_df = pd.read_csv(os.path.join(os.getcwd(),"data\\test.csv"))

    # Error handling when creating the engine to SQLite
    try:
        engine = create_engine('sqlite:///DLMDSPWP01.db', echo=False)
    except:
        print("Failed to create engine.")

    # Create new table in SQLite based on dataframe
    train_df.to_sql('train',con=engine, index=False, if_exists='replace')
    ideal_df.to_sql('ideal',con=engine, index=False, if_exists='replace') 

    # Split into 2 arrays for training
    x_train_data_quadratic, y_train_data_quadratic = (train_df["x"].values, train_df["y1"].values)
    x_train_data_linear, y_train_data_linear = (train_df["x"].values, train_df["y2"].values)
    x_train_data_sigmoid, y_train_data_sigmoid = (train_df["x"].values, train_df["y3"].values)
    x_train_data_cubic, y_train_data_cubic = (train_df["x"].values, train_df["y4"].values)
    #x_test_data, y_test_data = (test_df["x"].values, test_df["y"].values)
    
    # Initial guess
    init_guess_1 = [1.0]
    init_guess_2 = [1.0, 1.0]
    init_guess_3 = [1.0, 1.0, 1.0]
    init_guess_4 = [1.0, 1.0, 1.0, 1.0]

    # Find the best fit
    popt_train_quadratic, pcov_train_quadratic = curve_fit(quadratic, x_train_data_quadratic, y_train_data_quadratic, init_guess_1)
    popt_train_linear, pcov_train_linear = curve_fit(linear, x_train_data_linear, y_train_data_linear, init_guess_2)
    popt_train_sigmoid, pcov_train_sigmoid = curve_fit(sigmoid, x_train_data_sigmoid, y_train_data_sigmoid, init_guess_1)
    popt_train_cubic, pcov_train_cubic = curve_fit(cubic, x_train_data_cubic, y_train_data_cubic, init_guess_4)

    # New dataframe for RMSE
    RMSE_df = pd.DataFrame(columns=["Model", "RMSE"])
    result_df = pd.DataFrame(columns=["x", "y", "delta_y", "ideal_function"])
    result_df2 = pd.DataFrame(columns=["x", "y", "y_hat", "delta_y", "ideal_function"])
    result_df3 = pd.DataFrame(columns=["x", "y", 
        "y_hat_1", "delta_y_1", "RMSE_1", 
        "y_hat_2", "delta_y_2", "RMSE_2",
        "y_hat_3", "delta_y_3", "RMSE_3"])

    # Get item value in the test data 
    for index, row in test_df.iterrows():
        y_hat_quadratic = predict("quadratic", row['x'], popt_train_quadratic)
        y_hat_linear = predict("linear", row['x'], popt_train_linear)
        y_hat_sigmoid = predict("sigmoid", row['x'], popt_train_sigmoid)
        y_hat_cubic = predict("cubic", row['x'], popt_train_cubic)

        quadratic_delta = row['y'] - y_hat_quadratic
        linear_delta = row['y'] - y_hat_linear
        sigmoid_delta = row['y'] - y_hat_sigmoid
        cubic_delta = row['y'] - y_hat_cubic
        #print("Value of quadratic_delta={}, linear_delta={}, cubic_delta={}".format(quadratic_delta, linear_delta, cubic_delta))

        quadratic_RMSE = evaluation(y_hat_quadratic, row['y'])
        linear_RMSE = evaluation(y_hat_linear, row['y'])
        sigmoid_RMSE = evaluation(y_hat_sigmoid, row['y'])
        cubic_RMSE = evaluation(y_hat_cubic, row['y'])

        RMSE_dict = {
            'Quadratic': [quadratic_RMSE, row['x'], row['y'], quadratic_delta], 
            'Linear': [linear_RMSE, row['x'], row['y'], linear_delta], 
            'Cubic': [cubic_RMSE, row['x'], row['y'], cubic_delta]}

        RMSE_dict2 = {
            'Quadratic': [quadratic_RMSE, row['x'], row['y'], y_hat_quadratic, quadratic_delta], 
            'Linear': [linear_RMSE, row['x'], row['y'], y_hat_linear, linear_delta], 
            'Sigmoid': [sigmoid_RMSE, row['x'], row['y'], y_hat_sigmoid, sigmoid_delta], 
            'Cubic': [cubic_RMSE, row['x'], row['y'], y_hat_cubic, cubic_delta]}

        result_log = {'x': row['x'], 'y': row['y'], 
            'y_hat_1': y_hat_quadratic, 'delta_y_1': quadratic_delta, 'RMSE_1': quadratic_RMSE,
            'y_hat_2': y_hat_linear, 'delta_y_2': linear_delta, 'RMSE_2': linear_RMSE,
            'y_hat_3': y_hat_sigmoid, 'delta_y_3': sigmoid_delta, 'RMSE_3': sigmoid_RMSE,
            'y_hat_4': y_hat_cubic, 'delta_y_4': cubic_delta, 'RMSE_4': cubic_RMSE}

        #result_df = result_df.append(get_smallest_RMSE(RMSE_dict), ignore_index=True)
        result_df2 = result_df2.append(get_smallest_RMSE2(RMSE_dict2), ignore_index=True)
        #result_df3 = result_df3.append(result_log, ignore_index=True)
    
        # Create new table in SQLite based on dataframe
        #result_df.to_sql('result',con=engine, index=False, if_exists='replace')
        result_df2.to_sql('result2',con=engine, index=False, if_exists='replace')
        #result_df3.to_sql('temp_result',con=engine, index=False, if_exists='replace')
        #result_df3.to_csv(os.path.join(os.getcwd(),"data\\temp_result.csv"))

if __name__ == "__main__":
    main()