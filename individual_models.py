import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def individual_idents(lst):

    identifiers = list(set(df2['identifier'].tolist()))
    ident = df2['identifier'].tolist()

    '''fig, axis = plt.subplots(2, 2)
    fig.suptitle("Predictive Models Based on Identifier", fontsize=8, y=0.95)'''

    for iden in idents:

        if (ident.count(iden)) > 20:
            df = df2
            df = df.loc[df['identifier'] == iden]

            #setting the index of the graph as the date so that there is a regression
            # over the dates
            df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

            #print(df.head(10))

            #df.plot.line(y='target', x = 'date')
            #plt.show()

            predictors = ['market_cap',	'sector', 'index_membership', 'factor_1',
                      'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
                      'factor_7', 'factor_8', 'factor_9', 'factor_10']

            model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

            #print(df.head(5))

            train, test = train_test_split(df, test_size=0.3)

            model = LinearRegression()
            model.fit(train[predictors], train["target"])

            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index)

            r = r2_score(test["target"], preds)

            combined = pd.concat({"target": test["target"],"Predictions": preds}, axis=1)
            combined.plot()
            plt.title(iden + " with r^2: " + str(r))
            #a = ax[n][0]
            #b = ax[n][1]
            #axis[a, b].plot(combined)
            #axis[a, b].set_title(iden)
            #plt.title(iden)
            #plt.plot()

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
    path_to_file = "/Users/beadeoliveira/Desktop/data.csv"

    # df = pd.read_csv(path_to_file, index_col='identifier').T
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