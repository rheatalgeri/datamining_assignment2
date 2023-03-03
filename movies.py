import csv
from collections import defaultdict
from apyori import apriori
import time
# Step 1: Read the u.data file using a CSV reader
ratings = defaultdict(list)
with open('C:\\Users\\rheat\\Desktop\\IT_Learning\\Masters\\data_mining\\assignments\\assignment_2\\datasets\\movies\\u.data') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        user_id, movie_id, rating, timestamp = row
        ratings[user_id].append(movie_id)

# Step 2: Convert the data into a transactional format
transactions = list(ratings.values())

# Step 3: Convert the dataframe into a list of lists
transactions = [[str(item) for item in transaction] for transaction in transactions]



st = time.time()
# Step 4: Run the Apriori algorithm
results = list(apriori(transactions, min_support=0.1, min_confidence=0.5))
for result in results:
    print(result)

et = time.time()

time_taken = et - st

print("the time taken to run this algorithm is ", time_taken)
