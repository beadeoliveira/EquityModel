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


# DESCRIPTION: This model will run a regression for individual idenfitiers
# which are specified by the user. It functions similarly to the full model,
# but is constrained to each specified identifier


# Creates a function which take a list of identifiers and creates a model for
# each
def individual_idents(lst):

    for iden in lst:

        df = df2
        df = df.loc[df['identifier'] == iden]

        # Sets the index of the graph as the date so that the regression occurs
        # over the dates
        df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

        # Sets the  predictor values
        predictors = ['market_cap',	'sector', 'index_membership', 'factor_1',
                  'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
                  'factor_7', 'factor_8', 'factor_9', 'factor_10']

        # Uses the train_test_split to randomly select 30% of the data as
        # testing data and saving the rest for the creation/training of the
        # model
        train, test = train_test_split(df, test_size=0.3)

        # Defines the model as a Linear Regression
        model = LinearRegression()

        # Fits the model using the predictors above and defined target
        # training data
        model.fit(train[predictors], train["target"])

        # Creates the predictors using the model.predict
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        # Calculates the r^2 score based on the test values and the predicted
        # values
        r = r2_score(test["target"], preds)
        combined = pd.concat({"target": test["target"],"Predictions": preds}, axis=1)
        combined.plot()
        plt.title(iden + " with r^2: " + str(r))

        print("r2 of " + iden + " =" + str(r))

        y = df['target']
        X = df[['market_cap', 'sector', 'index_membership', 'factor_1',
                'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
                'factor_7', 'factor_8', 'factor_9', 'factor_10']]

        cv = KFold(n_splits=10, random_state=1, shuffle=True)

        modl = LinearRegression()

        scores = cross_val_score(modl, X, y,
                                 scoring='neg_mean_absolute_error',
                                 cv=cv, n_jobs=-1)

        # the lower the RMSE the better
        print('root mean squared error (RMSE) = ' + str(
            np.sqrt(np.mean(np.absolute(
                scores)))))

    plt.show()

if __name__ == '__main__':

    # Path to the given CSV containing the data
    # TODO: insert the path to the dataset that you would like to analyze
    # In this instance, I have used the path to the file on my computer as an
    # example
    path_to_file = "/Users/beadeoliveira/Desktop/data.csv"

    # Reads the information contained in the CSV
    df2 = pd.read_csv(path_to_file)

    idents = []

    answer = 'Y'

    while (answer == 'Y'):
        n = input("What identifier would you like to model?")
        idents.append(n)
        answer = input(
            "Would you like to add another? - answer 'Y' for yes and "
            "'N' for no")

    individual_idents(idents)