import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("http://bit.ly/w-data")
print(data)
print(data.shape)

print(data.dtypes)
x = data['Hours']
y = data['Scores']

data.plot(x='Hours',y='Scores', style='o')
plt.show()
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=0.2,random_state=50)
Lr = LinearRegression()
Lr.fit(X_train,Y_train)
print("training Complete")
line = Lr.coef_*x+Lr.intercept_

plt.scatter(x, y)
plt.plot(x, line);
plt.show()

prediction = Lr.predict(X_test)
df = pd.DataFrame({'Actual':Y_test, 'Prediction':prediction})
print(df)

hours = 9.25
pred = Lr.predict([[hours]])
print("No. of Hours = {}".format(hours))
print("Prediction Score = {}".format(pred[0]))

'''

y_pred = pd.DataFrame(predicted_values)
df = pd.DataFrame(Y_test)
df = pd.concat([df,y_pred],axis=1)
df.columns=["Actual","Y_pred"]
'''

from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(Y_test, prediction))