import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the sushi dataset
items = pd.read_csv('.\\datasets\\sushi3-2016\\sushi3.idata', delimiter='\t', header=None, names=['id', 'name', 'style', 'majority', 'minority', 'oiliness', 'price', 'popularity', 'calories'])
users = pd.read_csv('.\\datasets\\sushi3-2016\\sushi3.udata', delimiter='\t', header=None, names=['id', 'age', 'gender', 'region', 'prefecture'])

# Load the preference order and score data for item set B
order_b = pd.read_csv('.\\datasets\\sushi3-2016\\sushi3b.5000.10.order', delimiter='\t', header=None, names=[f'item{i}' for i in range(1, 11)])
score_b = pd.read_csv('.\\datasets\\sushi3-2016\\sushi3b.5000.10.score', delimiter='\t', header=None, names=[f'item{i}' for i in range(1, 11)])

order_b.dropna(inplace=True)
score_b.dropna(inplace=True)
items.dropna(inplace=True)
users.dropna(inplace=True)
# Combine the preference order and score data for item set B
data_b = pd.concat([order_b, score_b], axis=1)

# Convert the preference order and score data for item set B to binary data
data_b = pd.get_dummies(data_b)

# Run Apriori algorithm to find frequent itemsets with minimum support of 0.05
frequent_itemsets = apriori(data_b, min_support=0.05, use_colnames=True)

# Generate association rules with minimum lift of 1.2
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Print the association rules
print(rules)
