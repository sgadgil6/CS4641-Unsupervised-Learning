import sys
import data_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis


def pca(data,target,title):
	X = data
	y = target
	target_names = ['benign','malignant'] if title=='cancer' else ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	# print("original data:\n", X)

	pca = PCA(n_components=10)
	X_r = pca.fit(X).transform(X)

	# Percentage of variance explained for each components
	print('explained variance ratio (first two components): %s'
		  % str(pca.explained_variance_ratio_))

	plt.figure()
	colors = ['navy', 'turquoise']
	lw = 2

	for color, i, target_name in zip(colors, [0, 1], target_names):
		plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
					label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of ' + title.title())

	plt.show()

	record(title+"_pca.csv", X_r, y)

	# For clustering dataset
	N = X_r[y==0].shape[0]
	X_0 = np.c_[ X_r[y==0], np.zeros(N) ] # Add a new column
	N = X_r[y==1].shape[0]
	X_1 = np.c_[ X_r[y==1], np.ones(N) ]
	X_r = np.r_[ X_0, X_1 ]
	record(title+"_pca_clu.csv", X_r,y)

def ica(data,target,title):
	X = data
	y = target
	# target_names = ['benign','malignant'] if title=='cancer' else ['normal', 'fraud']
	# print("original data:\n", X)

	ica = FastICA(n_components=X.shape[1], algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)
	X_r = ica.fit(X).transform(X)

	kurt = kurtosis(X_r)

	plt.figure()
	plt.plot(np.arange(kurt.shape[0]), kurt)
	plt.xlabel('ICA Feature')
	plt.ylabel('Kurtosis')
	plt.title(title.title() + '-ICA Kurtosis')
	plt.show()

	record(title+"_ica.csv", X_r, y)

	# For clustering dataset
	N = X_r[y==0].shape[0]
	X_0 = np.c_[ X_r[y==0], np.zeros(N) ] # Add a new column
	N = X_r[y==1].shape[0]
	X_1 = np.c_[ X_r[y==1], np.ones(N) ]
	X_r = np.r_[ X_0, X_1 ]
	record(title+"_ica_clu.csv", X_r,y)

def rp(data,target,title):
	X = data
	y = target
	target_names = ['benign','malignant'] if title=='cancer' else ['normal', 'fraud']

	rp = GaussianRandomProjection(n_components=10, eps=0.1, random_state=None)
	X_r = rp.fit(X).transform(X)

	record(title+"_rp.csv", X_r, y)

	# For clustering dataset
	N = X_r[y==0].shape[0]
	X_0 = np.c_[ X_r[y==0], np.zeros(N) ] # Add a new column
	N = X_r[y==1].shape[0]
	X_1 = np.c_[ X_r[y==1], np.ones(N) ]
	X_r = np.r_[ X_0, X_1 ]
	record(title+"_rp_clu.csv", X_r,y)

def lda(data,target,title):
	X = data
	y = target
	target_names = ['benign','malignant'] if title=='cancer' else ['normal', 'fraud']
	# print("original data:\n", X)

	lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
			  solver='svd', store_covariance=False, tol=0.0001)
	X_r = lda.fit(X, y).transform(X)

	# Percentage of variance explained for each components
	print('explained variance ratio: %s'
		  % str(lda.explained_variance_ratio_))

	record(title+"_lda.csv", X_r, y)

	# For clustering dataset
	N = X_r[y==0].shape[0]
	X_0 = np.c_[ X_r[y==0], np.zeros(N) ] # Add a new column
	N = X_r[y==1].shape[0]
	X_1 = np.c_[ X_r[y==1], np.ones(N) ]
	X_r = np.r_[ X_0, X_1 ]
	record(title+"_lda_clu.csv", X_r,y)


def record(filename, X_r, y):
	FOLDER = './data/'
	with open(FOLDER+filename, 'w') as fhand:
		episodes = np.c_[ X_r, y ] # add a column as the target
		for epi in episodes:
			line = ""
			for num in epi:
				line += "{:.3f},".format(num)
			line = line[:-1] + '\n' # get rid of the last comma
			fhand.write(line)

if __name__ == '__main__':
	filename = sys.argv[1]

	dataset = data_loader.load_data(filename) # argv[1] is the file storing the dataset
	title = filename[:-4]
	data = dataset[:,:-1]
	target = dataset[:,-1]
	pca(data,target,title)
	ica(data,target,title)
	rp(data,target,title)
	lda(data,target,title)
