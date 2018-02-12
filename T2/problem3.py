# Don't change these imports. Note that the last two are the
# class implementations that you will implement in
# LogisticRegression.py and GaussianNaiveBayes.py
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression
from GaussianGenerativeModel import GaussianGenerativeModel


## These are the hyperparameters to the classifiers. You may need to
# adjust these as you try to find the best fit for each classifier.

# Logistic Regression parameters
eta = .1
lambda_parameter = .1


# Do not change anything below this line!!
# -----------------------------------------------------------------

# Read from file and extract X and Y
df = pd.read_csv("fruit.csv")
X = df[['width', 'height']].values
Y = (df['fruit'] - 1).values

nb1 = GaussianGenerativeModel(isSharedCovariance=False)
nb1.fit(X,Y)
nb1.visualize("generative_result_separate_covariances.png")

nb2 = GaussianGenerativeModel(isSharedCovariance=True)
nb2.fit(X,Y)
nb2.visualize("generative_result_shared_covariances.png")

lr = LogisticRegression(eta=eta, lambda_parameter=lambda_parameter)
lr.fit(X,Y)
lr.visualize('logistic_regression_result.png')

X_test = np.array([[4,11],[8.5,7]])
Y_nb1 = nb1.predict(X_test)
Y_nb2 = nb2.predict(X_test)
Y_lr = lr.predict(X_test)

print("Test fruit predictions for Gaussian Model:")
print("width 4 cm and height 11 cm: " + str(Y_nb1[0]))
print("width 8.5 cm and height 7 cm: " + str(Y_nb1[1]))

print("Test fruit predictions for Shared Covariance Gaussian Model:")
print("width 4 cm and height 11 cm: " + str(Y_nb2[0]))
print("width 8.5 cm and height 7 cm: " + str(Y_nb2[1]))

print("Test fruit predictions for Linear Regression:")
print("width 4 cm and height 11 cm: " + str(Y_lr[0]))
print("width 8.5 cm and height 7 cm: " + str(Y_lr[1]))