import sys
import data_loader

import k_means
import ex_max

if __name__ == '__main__':
	filename = sys.argv[1]
	dataset = data_loader.load_data(filename) # argv[1] is the file storing the dataset
	# Km = k_means.K_Means(dataset,filename[:-4].title())
	# Km.run()
	Em = ex_max.ExpectationMaximization(dataset,filename[:-4].title())
	Em.run()