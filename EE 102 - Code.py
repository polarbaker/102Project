

# mom look no hands!

import numpy as np

from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

import pandas as pd

df = pd.read_csv('RealEstate.csv', sep=',', header=None)


col_count = len(df.columns)
row_count = len(df.index) - 2 # accomodate for the label

# Data Parsing
y = df.iloc[1:row_count,6] 	# PRICE [OUTPUT]
x_location = df.iloc[1:row_count,0]		# LOCATION [INPUT] 
x_bedrooms = df.iloc[1:row_count,1]		# BEDROOMS [INPUT]
x_bathrooms = df.iloc[1:row_count,2]		# BATHROOMS [INPUT]
x_size = df.iloc[1:row_count,3]			# SIZE [INPUT]
x_pricesqft = df.iloc[1:row_count,4]		# PRICE/SQ FT. [INPUT]
x_status = df.iloc[1:row_count,5]		# STATUS [INPUT]

'''
########         sets samples 
n_samples = 1000

n_outliers = 50

##########          sets our coefficients

X, y, coef = datasets.make_regression(n_samples=n_samples, 
                                      n_features=col_count,
                                      n_informative=col_count, 
                                      noise=10,
                                      coef=True,
                                      random_state=0)

#######            Add outlier data

#use a function from numpy to create a small bit of random variables 
np.random.seed(0)

#plots the outliers so we can influence the pretty lina

#X[:n_outliers] =  3 + 0.5 * np.random.normal(size = (n_outliers, 1))
#y[:n_outliers] = -3 + 10 * np.random.normal(size = n_outliers)


'''
X = x_size

# reshaping data
X = X.reshape(-1,1)
X = list(map(int, X))

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
#outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
# print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')

#plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
#            label='Outliers')

plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')

plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')

plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
