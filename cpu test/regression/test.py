from sklearn.neural_network import MLPRegressor
import pandas as pd
import time


print("Reading Data...")

data = pd.read_csv("cpu test\\regression\\data.csv")


X_base = data[['mimtemp', 'maxtemp', 'rainfall',
               'countDay-3', 'countDay-2', 'countDay-1']]
y_base = data['count']


print("Data Read Compleated, Starting Model Traning...")
start_time = time.time()

mlpr = MLPRegressor()

mlpr.fit(X_base, y_base)

finish_time = time.time()

print('Traning was completed.')
print('Total Time Was: ' + str(((finish_time - start_time))) + ' Seconds')
print('Total Time Was: ' + str(((finish_time - start_time) / 60)) + ' Minutes')
print('Total Time Was: ' + str(((finish_time - start_time) / 60 / 60)) + ' Hours')
