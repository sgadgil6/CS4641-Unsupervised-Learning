import sys
import data_loader
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib.ticker import MaxNLocator


class ExpectationMaximization():
    def __init__(self,dataset,title):
        self.title = title
        self.data = scale(dataset[:,:-1])
        self.labels = dataset[:,-1]
        self.sample_size = 300

        self.target_cluster = 2
        self.clusters = []
        for i in range(1, 20, 2):
            self.clusters.append(i)
        self.running_times = []
        self.model_scores = []
        self.homogeneity_scores = []
        self.completeness_scores = []
        # self.v_measure_scores = []
        # self.adjusted_rand_scores = []
        # self.adjusted_mutual_info_scores = []
        self.aics = []
        self.bics = []
        # self.silhouette_scores = []

    def run(self):
        for n in self.clusters:
            self.fit_ExpectationMaximization(GaussianMixture(n_components=n))
        self.plot()

    def fit_ExpectationMaximization(self,estimator):
        start = time()
        estimator.fit(self.data)
        predictions = estimator.predict(self.data)
        # Gather informations
        self.running_times.append( time()-start )
        self.model_scores.append( estimator.score(self.data, predictions) )
        self.homogeneity_scores.append( metrics.homogeneity_score(self.labels, predictions) )
        self.completeness_scores.append( metrics.completeness_score(self.labels,predictions) )
        self.aics.append( estimator.aic(self.data) )
        self.bics.append( estimator.bic(self.data) )
        # self.v_measure_scores.append( metrics.v_measure_score(self.labels, estimator.labels_) )
        # self.adjusted_rand_scores.append( metrics.adjusted_rand_score(self.labels, estimator.labels_) )
        # self.adjusted_mutual_info_scores.append( metrics.adjusted_mutual_info_score(self.labels,  estimator.labels_) )
        # self.silhouette_scores.append( metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=self.sample_size) )

    def plot(self):

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.subplot(2,2,1)
        plt.plot(self.clusters, self.running_times)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Running Time')
        plt.title(self.title + '-EM-Running Time')

        plt.subplot(2,2,2)
        plt.plot(self.clusters, self.model_scores)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Model Scores')
        plt.title(self.title + '-EM-Log Probabilities')


        # plt.subplot(2,3,3)
        # plt.plot(self.clusters, self.homogeneity_scores)
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Homogeneity Scores')
        # plt.title(self.title + '-EM-Homogeneity')
        #
        # plt.subplot(2,3,4)
        # plt.plot(self.clusters, self.completeness_scores)
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Completeness')
        # plt.title(self.title + '-EM-Completeness')
        #
        # plt.subplot(2,3,5)
        # plt.plot(self.clusters, self.aics)
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('AIC Scores')
        # plt.title(self.title + '-EM-AIC')
        #
        # plt.subplot(2,3,6)
        # plt.plot(self.clusters, self.bics)
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('BIC Scores')
        # plt.title(self.title + '-EM-BIC')

        plt.show()


if __name__ == '__main__':
    filename = sys.argv[1]
    dataset = data_loader.load_data(filename) # argv[1] is the file storing the dataset
    project = ExpectationMaximization(dataset,filename[:-4].title())
    project.run()