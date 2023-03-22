import pandas as pd
from collections import defaultdict
from apyori import apriori
import time

dataset = pd.read_csv('C:\\Users\\rheat\\Desktop\\IT_Learning\\Masters\\data_mining\\assignments\\assignment_2\\datasets\\groceries\\groceries1.csv',sep=',')

transactions = []
# convert the data into a transactional format
for i in range(0, 4999):
    transactions.append([str(dataset.values[i,u]) for u in range(1, 33)])

st = time.time()
# run the apriori algorithm

rules = apriori(transactions, min_support=0.0022, min_confidence=0.20, min_lift=3, min_length = 2)
results = list(rules)

et = time.time()

time_taken = et - st

print("the time taken to run this algorithm is ", time_taken)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))

print(results_list)