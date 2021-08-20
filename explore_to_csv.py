import numpy as np
import params as p
import pandas
from DL import estimator

world_matrix = p.world_object

# This file would from the world grid world to create area of different values
# We can set threshold for humid and temp (wall is binary already)

# load test dataset
df_test = pandas.read_csv("./training_data.csv", header=0)
test_set = df_test.values
X_test = test_set[:5, 0:3].astype(float)
Y_test = test_set[:, 3]

a = estimator.predict(X_test)

print(a)

