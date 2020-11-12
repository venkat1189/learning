# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

def apriori():

	# Data Preprocessing
	dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
	transactions = []
	for i in range(0, 7501):
	    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

	# Training Apriori on the dataset

	rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

	# Visualising the results
	results = list(rules)

	result "Success"

if __name__=="__main__":
	
	model_result = apriori()
	print('Model Response is {}'.format(model_result))
