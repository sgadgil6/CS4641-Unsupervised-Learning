import sys
import numpy as np
import random
import data_loader
#from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier	# For Neural Network
import matplotlib.pyplot as plt		# For plotting the data
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

EPISODE_NUM = 10
TITLE = "Learning Curve (Neural Network)-"

def neural_network(dataset,filename):
	numOfFeature = dataset.shape[1]-1
	X = dataset[:,0:numOfFeature]
	y = dataset[:,numOfFeature]
	X = preprocess_data(X)
	X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=EPISODE_NUM)
	# X_train,X_test = preprocess_feature(X_train,X_test)

	model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
		beta_1=0.9, beta_2=0.999, early_stopping=False,
		epsilon=1e-08, hidden_layer_sizes=(55,55,55,55,55), learning_rate='constant',
		learning_rate_init=0.01, max_iter=200, momentum=0.9,
		nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
		warm_start=False)
	label = filename
	learn_cur(model,TITLE,X,y,label)

	print "Check out the graph popped up"
	plt.show()

def preprocess_data(X):
	scaler = StandardScaler().fit(X)
	return scaler.transform(X)

def preprocess_feature(X_train,X_test):
	# Preprocess the Features
	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)	# Rescale the data
	X_test = scaler.transform(X_test)
	return X_train,X_test


def score_model(model,X,y):
	cv_scores = cross_val_score(model,X,y,cv=EPISODE_NUM)
	print "Cross Validation Scores:"
	print cv_scores


def fit_model(model,X_train,y_train,X_test,y_test):
	print "Training size:\t"
	print X_train.shape[0]
	model.fit(X_train,y_train)
	accu = accuracy_score(y_test,model.predict(X_test))
	print "Accuracy: \ t"
	print accu

	return accu


def learn_cur(model, title, X, y, label=None, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
								# ylim : tuple, shape (ymin, ymax), optional. Defines minimum and maximum yvalues plotted.

	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

	plt.figure()
	plt.title(title + label)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples ")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					train_scores_mean + train_scores_std, alpha=0.1,
					color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")

	return plt

if __name__ == '__main__':
	dataset = data_loader.load_data(sys.argv[1]) # argv[1] is the file storing the dataset
	filename = sys.argv[1][:-4].title()
	neural_network(dataset,filename)