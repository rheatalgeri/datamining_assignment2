from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
# create a list of transactions from the dataset
dataset = []
with open('C:\\Users\\rheat\\Desktop\\IT_Learning\\Masters\\data_mining\\assignments\\assignment_2\\datasets\\chess\\kr-vs-kp.data', 'r') as file:
    for line in file:
        transaction = line.strip().split(',')
        dataset.append(transaction)

# perform one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# print the results
print(frequent_itemsets)