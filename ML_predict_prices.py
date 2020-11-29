import pandas
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('iphone_price.csv')
print(data)
# code to show data in scatter chart
plot.scatter(data['version'], data['price'])
plot.show()
# code to show data in bar graph
plot.bar(data['version'], data['price'])
plot.show()
# code to predict future data
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[12]]))
print(model.predict([[20]]))