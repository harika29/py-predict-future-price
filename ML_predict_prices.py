import pandas
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('iphone_price.csv')
print(data)
plot.scatter(data['version'], data['price'])
plot.show()
plot.bar(data['version'], data['price'])
plot.show()
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[12]]))
print(model.predict([[20]]))