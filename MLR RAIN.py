import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split
  
# read the cleaned data 
data = pd.read_csv("austin_final.csv") 
  
# the features or the 'x' values of the data 
# these columns are used to train the model 
# the last column, i.e, precipitation column  
# will serve as the label  
X = data.drop(['PrecipitationSumInches'], axis = 1) 
  
# the output or the label. 
Y = data['PrecipitationSumInches'] 
# reshaping it into a 2-D vector 
Y = Y.values.reshape(-1, 1) 
  
# consider a random day in the dataset 
# we shall plot a graph and observe this 
# day 
day_index = 798
days = [i for i in range(Y.size)] 
  
# initialize a linear regression classifier 
clf = LinearRegression() 
# train the classifier with our  
# input data. 
clf.fit(X, Y) 
#print(clf.intercept_)
#print(clf.coef_)

# give a sample input to test our model 
# this is a 2-D vector that contains values 
# for each column in the dataset. 
inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45], 
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]]) 
inp = inp.reshape(1, -1) 
output = clf.predict(inp)
# print the output. 

# plot a graph of the precipitation levels 
# versus the total number of days. 
# one day, which is in red, is 
# tracked here. It has a precipitation 
# of approx. 2 inches. 
print("the precipitation trend graph: ") 
plt.scatter(days, Y, color = 'g') 
plt.scatter(days[day_index], Y[day_index], color ='r') 
plt.title("Precipitation level") 
plt.xlabel("Days") 
plt.ylabel("Precipitation in inches") 

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
lm2 = LinearRegression()

# Fit Model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# RMSE
print("mean squared error is",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("mean absolute error is",metrics.mean_absolute_error(y_test, y_pred))

print('The precipitation in inches for the input is:', output) 
print('The precipitation in inches for the input is:', (output*25.4)) 

'''import statsmodels.formula.api as smf
lm1 = smf.ols(formula='Y ~ X', data=data).fit()

# print the coefficients
lm1.params
lm1.summary()

'''  
'''plt.show() 
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 
                  'WindAvgMPH'], axis = 1) 
  
print("Precipitation vs selected attributes graph: ") 
  
for i in range(x_vis.columns.size): 
    plt.subplot(3, 2, i + 1) 
    plt.scatter(days, x_vis[x_vis.columns.values[i][:100]], 
                                               color = 'g') 
  
    plt.scatter(days[day_index],  
                x_vis[x_vis.columns.values[i]][day_index], 
                color ='r') 
  
    plt.title(x_vis.columns.values[i]) 
  
plt.show() '''

