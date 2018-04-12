# Kaggle-House-Prices
This is my second project on Kaggle. Here is a log about the project I do every time.

2018.4.11
(1) Show data and get some information about the distribution of data.

(2) Split train data into several folds and make cross-validation, code as follows:

#x_train, x_test, y_train, y_test = train_test_split(train[predictors], target, random_state=42, test_size=.2)

2018.4.12
(1) Fix the norm -> expo about the final prediction result

(2) One hot encoding for non-numeric attributes, there are some difference between only two or two more values.
If there are only two values for the attribute, use code as follows:

#combined['enc_Street'] = pd.get_dummies(combined.Street, drop_first=True)

However, if there are more than two values, we need to use code as follows:

#GarageQual_dummies = pd.get_dummies(combined['GarageQual'], prefix='GarageQual')
#combined = pd.concat([combined,GarageQual_dummies], axis=1)
#combined.drop('GarageQual', axis=1, inplace=True)

Got score 9.49317

(3) Find some abnormal points, and drop them. usually we first need to use plot to show figure about the data.

(4) Observe the correlation between attributes and the target, we can know some important attributes or drop some irrelevant attributes.
some codes as follows:

#numeric_features = hp_train.select_dtypes(include=[np.number])
#corr = numeric_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])

Got score 0.16408 (Top62%)