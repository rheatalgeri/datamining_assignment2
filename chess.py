from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
import time
# create a list of transactions from the dataset
dataset = []
with open('.\\datasets\\chess\\kr-vs-kp.data', 'r') as file:
    for line in file:
        transaction = line.strip().split(',')
        dataset.append(transaction)

# perform one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

start_time = time.time()
# apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

end_time = time.time()

time_taken = end_time - start_time

print("the time taken to run this algorithm is ", time_taken)