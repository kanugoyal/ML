import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
data = pd.read_csv('iphone_price.csv')
plt.scatter(data['version'],data['price'])
plt.show()
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[30]]))