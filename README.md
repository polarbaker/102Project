

# mom look no hands!

import numpy as np

from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

import csv

RealEstatePrice = []
RealEstateSize = []

#opens csv and inputs it into f
#then inputs that data into a list by type casting list reader

with open('RealEstate.csv','rt') as f:
    reader = csv.reader(f, delimiter = ',')
    for row in reader:
        RealEstatePrice.append(row[2])
        RealEstateSize.append(row[5])
    


########         sets samples 
n_samples = 781

n_outliers = 50

##########          sets our coefficients

X, y, coef = datasets.make_regression(n_samples=n_samples, 
                                      n_features=1,
                                      n_informative=1, 
                                      noise=10,
                                      coef=True,
                                      random_state=0)

#######            Add outlier data

#use a function from numpy to create a small bit of random variables 
np.random.seed(0)

#plots the outliers so we can influence the pretty lina

#X[:n_outliers] =  3 + 0.5 * np.random.normal(size = (n_outliers, 1))
#y[:n_outliers] = -3 + 10 * np.random.normal(size = n_outliers)
X = RealEstateSize
y = RealEstatePrice

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y, None)

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
print(coef, lr.coef_, ransac.estimator_.coef_)

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
