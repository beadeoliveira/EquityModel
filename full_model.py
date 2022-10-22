import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# DESCRIPTION: This model will run a regression on all of the data over the
# time period given, treating identifier, market cap, the factors, etc as
# independent variables. The regression is completed using SKLearn which
# utilizes test and training data to fit a learned model to the data.

def calculate_model(data):
    # Sets the index of the graph as the date so that the regression occurs
    # over the dates
    data.set_index(pd.DatetimeIndex(data['date']), inplace=True)

    # Sets the  predictor values
    predictors = ['market_cap', 'sector', 'index_membership', 'factor_1',
                  'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
                  'factor_7', 'factor_8', 'factor_9', 'factor_10']

    # Uses the train_test_split to randomly select 30% of the data as testing
    # data and saving the rest for the creation/training of the model
    train, test = train_test_split(data, test_size=0.3)

    # Defines the model as a Linear Regression
    model = LinearRegression()

    # Fits the model using the predictors above and defined target training data
    model.fit(train[predictors], train["target"])

    # Creates the predictors using the model.predict
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)

    # Calculates the r^2 score based on the test values and the predicted values
    r = r2_score(test["target"], preds)

    combined = pd.concat({"target": test["target"], "Predictions": preds},
                         axis=1)
    combined.plot()
    plt.title("All of them Together")
    plt.show()

    # k_fold test
    # predictor (x) and response variables (y)
    y = data['target']
    X = data[['market_cap', 'sector', 'index_membership', 'factor_1',
            'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
            'factor_7', 'factor_8', 'factor_9', 'factor_10']]

    # Conducts the K_Fold test
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # modl = LinearRegression()

    # Calculates the cross validation score which is later used to calculate the
    # RMSE below
    scores = cross_val_score(model, X, y,
                             scoring='neg_mean_absolute_error',
                             cv=cv, n_jobs=-1)

    # the lower the RMSE the better
    print("r^2 is = " + str(r))
    print('root mean squared error (RMSE) = ' + str(np.sqrt(np.mean(np.absolute(
        scores)))))


if __name__ == '__main__':

    # Path to the given CSV containing the data
    # TODO: insert the path to the dataset that you would like to analyze
    # In this instance, I have used the path to the file on my computer as an
    # example
    path_to_file = "/Users/beadeoliveira/Desktop/data.csv"

    # Reads the information contained in the CSV
    df = pd.read_csv(path_to_file)

    calculate_model(df)

    print('\n Model Complete')
