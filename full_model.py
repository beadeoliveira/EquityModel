import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Path to the given CSV containing the data
path_to_file = "/Users/beadeoliveira/Desktop/data.csv"

# df = pd.read_csv(path_to_file, index_col='identifier').T
df = pd.read_csv(path_to_file)

# setting the index of the graph as the date so that there is a regression
# over the dates
df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

# df = df[df['identifier'] == 'PEOLTD6JT1H8']

predictors = ['market_cap', 'sector', 'index_membership', 'factor_1',
              'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
              'factor_7', 'factor_8', 'factor_9', 'factor_10']

model = RandomForestClassifier(n_estimators=100, min_samples_split=200,
                               random_state=1)

'''lab = preprocessing.LabelEncoder()
df['target'] = lab.fit_transform(df['target'])'''

print(df.head(5))

train, test = train_test_split(df, test_size=0.3)

model = LinearRegression()
model.fit(train[predictors], train["target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

r = r2_score(test["target"], preds)

print(r)

combined = pd.concat({"target": test["target"], "Predictions": preds}, axis=1)
combined.plot()
plt.title("All of them Together")
# plt.show()

# Now do k_fold test
# predictor (x) and response variables (y)
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
print('root mean squared error (RMSE) = ' + str(np.sqrt(np.mean(np.absolute(
    scores)))))

if __name__ == '__main__':
    print('done')
