# Kaggle-House-Prices
This is my second project on Kaggle. Here is a log about the project I do every time.

2018.4.11

(1) Show data and get some information about the distribution of data.

(2) Split train data into several folds and make cross-validation, code as follows:

#x_train, x_test, y_train, y_test = train_test_split(train[predictors], target, random_state=42, test_size=.2)

Got score 9.49317

2018.4.12

(1) Fix the norm -> expo about the final prediction result

(2) One hot encoding for non-numeric attributes, there are some difference between only two or two more values.
If there are only two values for the attribute, use code as follows:

#combined['enc_Street'] = pd.get_dummies(combined.Street, drop_first=True)

However, if there are more than two values, we need to use code as follows:

#GarageQual_dummies = pd.get_dummies(combined['GarageQual'], prefix='GarageQual')
#combined = pd.concat([combined,GarageQual_dummies], axis=1)
#combined.drop('GarageQual', axis=1, inplace=True)

(3) Find some abnormal points, and drop them. usually we first need to use plot to show figure about the data.

(4) Observe the correlation between attributes and the target, we can know some important attributes or drop some irrelevant attributes.
some codes as follows:

#numeric_features = hp_train.select_dtypes(include=[np.number])
#corr = numeric_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])

Got score 0.16408 (Top62%)

2018.4.15

(1) Preprocess most 15 relevant attributes sequentially.

(2) For numerical attributes, if not empty, add as predictors. Otherwise, fill up median, and then add as predictors.
However, for non-numerical attributes, if not empty, one-hot(2 attributes) or get dummy (multiple attributes). Otherwise, fill up at the most value,
and then one-hot as predictors.

Got score 0.15396 (Top55%)

2018.4.16

(1) Preprocess all relevant attributes of numerical sequentially. (total 38 attributes)

(2) Try to use combined model, now have tried LinearRegression() and RidgeCV(). However, do not get better results significantly.

Got score 0.13223 (Top34%)

2018.4.17

(1) because there are a large number of codes, divide the preprocessing into a new .py file called preprocessdata.

#import sys
#sys.path.append("preprocessdata")
#import preprocessdata as pre

(2) begin to use non-numerical attributes. however, feel a little dizzy about handling too many non-numerical attributes. the grade improves a little

Got score 0.13133 (Top32%)
