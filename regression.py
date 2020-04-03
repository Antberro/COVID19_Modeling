import pandas as pd
import matplotlib.pyplot as plt 
import os
from math import log, e, exp
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

def import_daily_data(country=None):
	"""
	Imports data and returns data for country arg grouped by day.
	"""
	path = os.path.join("novel-corona-virus-2019-dataset", "covid_19_data.csv")
	data = pd.read_csv(path)
	if not country:
		data = data.groupby('ObservationDate').sum()
		return data
	else:
		in_country = data['Country/Region'] == country
		country_data = data[in_country]
		country_data = country_data.groupby('ObservationDate').sum()
		return country_data

def split_data(data, label_type, train_ratio=0.6):
	"""
	Splits data into train and test data with label_type as labels.
	The ratio to train data to all data is given by train_ratio.
	"""
	n, _ = data.shape
	cutoff = int(train_ratio * n)
	X_train = np.array([range(cutoff)]).T 
	train_labels = np.array([data.iloc[0:cutoff, :][label_type]]).T
	X_test = np.array([range(cutoff, n)]).T 
	test_labels = np.array([data.iloc[cutoff:, :][label_type]]).T

	return X_train, train_labels, X_test, test_labels
	
def log_scale(col_vector):
	"""
	Given data in col_vector, take logarithm of each element.
	"""
	output = [None for _ in range(col_vector.shape[0])]
	for i in range(col_vector.shape[0]):
		if col_vector[i,0] > 0:
			output[i] = log(col_vector[i,0])
		else:
			output[i] = col_vector[i,0]
	return np.array([output]).T

def linear_regression(X_train, Y_train, xval=None) :
	"""
	Create linear regression model on data X with labels Y.
	"""
	if not xval:
		model = LinearRegression(fit_intercept=False).fit(X_train, log_scale(Y_train))
	else:
		model = LinearRegression(fit_intercept=False)
		results = cross_validate(model, X_train, log_scale(Y_train), cv=xval, return_estimator=True)
		coefs = np.array([float(m.coef_) for m in results['estimator']])
		avg_coef = np.mean(coefs)
		model.coef_ = np.array([[avg_coef]])
		model.intercept_ = 0
	return model

def get_error(model, X, Y):
	"""
	Calculates error using squared loss.
	"""
	G = model.predict(X)
	return np.mean((G - log_scale(Y))**2, axis=0)

def visualize_model(model, X, Y, linear=True):
	"""
	Plots data X,Y along with the regression model.
	"""
	if linear:
		plt.scatter(X, log_scale(Y), s=10, c='red')
		plt.plot(X, model.predict(X), c='blue')
		plt.xlabel("Days Since ")
		plt.ylabel("log(Feature)")
	else:
		plt.scatter(X, Y, s=10, c='red')
		plt.plot(X, np.exp(model.coef_ * X), c='blue')
		plt.xlabel("Days Since ")
		plt.ylabel("Feature")

	plt.show()


if __name__ == "__main__":
	usa_data = import_daily_data('US')
	X_train, Y_train, X_test, Y_test = split_data(usa_data, 'Confirmed', 0.8)

	X = np.vstack((X_train, X_test))
	Y = np.vstack((Y_train, Y_test))

	model = linear_regression(X_train, Y_train)
	print("training error: ", get_error(model, X_train, Y_train))
	print("test error: ", get_error(model, X_test, Y_test))
	visualize_model(model, X, Y, linear=False)

	
	