import sys
import data_loader
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib.ticker import MaxNLocator


class K_Means():
	def __init__(self,dataset,title):
		self.data = scale(dataset[:,:-1])
		self.title = title
		self.labels = dataset[:,-1]
		self.sample_size = 300

		self.clusters = []
		for i in range(1, 25, 2):
			self.clusters.append(i)
		self.running_times = []
		self.inertia = []
		self.homogeneity_scores = []
		self.completeness_scores = []
		self.v_measure_scores = []
		self.adjusted_rand_scores = []
		self.adjusted_mutual_info_scores = []
		self.silhouette_scores = []

	def run(self):
		for n in self.clusters:
			self.fit_k_means(KMeans(init='k-means++', n_clusters=n, n_init=10))
			# fit_k_means(KMeans(init='random', n_clusters=n, n_init=10))
		self.plot()



	def fit_k_means(self,estimator):
		start = time()
		estimator.fit(self.data)
		# Gather informations
		self.running_times.append( time()-start )
		self.inertia.append( estimator.inertia_ )
		self.homogeneity_scores.append( metrics.homogeneity_score(self.labels, estimator.labels_) )
		self.completeness_scores.append( metrics.completeness_score(self.labels, estimator.labels_) )
		self.v_measure_scores.append( metrics.v_measure_score(self.labels, estimator.labels_) )
		self.adjusted_rand_scores.append( metrics.adjusted_rand_score(self.labels, estimator.labels_) )
		self.adjusted_mutual_info_scores.append( metrics.adjusted_mutual_info_score(self.labels,  estimator.labels_) )
		# self.silhouette_scores.append( metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=self.sample_size) )

	def plot(self):

		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

		plt.subplot(2,2,1)
		plt.plot(self.clusters, self.running_times)
		plt.xlabel('Number of Clusters')
		plt.ylabel('Running Time')
		plt.title(self.title + '-KM-Running Time')

		plt.subplot(2,2,2)
		plt.plot(self.clusters, self.inertia)
		plt.xlabel('Number of Clusters')
		plt.ylabel('Model Inertia')
		plt.title(self.title + '-KM-Model Inertia')


		# plt.subplot(2,4,3)
		# plt.plot(self.clusters, self.homogeneity_scores)
		# plt.xlabel('Number of Clusters')
		# plt.ylabel('Homogeneity Scores')
		# plt.title(self.title + '-KM-Homogeneity')
        #
		# plt.subplot(2,4,4)
		# plt.plot(self.clusters, self.completeness_scores)
		# plt.xlabel('Number of Clusters')
		# plt.ylabel('Completeness')
		# plt.title(self.title + '-KM-Completeness')
        #
		# plt.subplot(2,4,5)
		# plt.plot(self.clusters, self.v_measure_scores)
		# plt.xlabel('Number of Clusters')
		# plt.ylabel('V Measure Scores')
		# plt.title(self.title + '-KM-V Measure')
        #
		# plt.subplot(2,4,6)
		# plt.plot(self.clusters, self.adjusted_rand_scores)
		# plt.xlabel('Number of Clusters')
		# plt.ylabel('Rand Scores')
		# plt.title(self.title + '-KM-Rand Scores')
        #
		# plt.subplot(2,4,8)
		# plt.plot(self.clusters, self.adjusted_mutual_info_scores)
		# plt.xlabel('Number of Clusters')
		# plt.ylabel('Mutual Information Scores')
		# plt.title(self.title + '-KM-Mutual Information')

		plt.show()




if __name__ == '__main__':
	filename = sys.argv[1]
	dataset = data_loader.load_data(filename) # argv[1] is the file storing the dataset
	project = K_Means(dataset,filename[:-4].title())
	project.run()