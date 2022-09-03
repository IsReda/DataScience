#Import pandas
import pandas as pd
#Import predection model
from sklearn.tree import DecisionTreeRegressor
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split
#Import mean absolute error
from sklearn.metrics import mean_absolute_error

#Data path
iowa_data_path = "C:\\Users\\redak\\OneDrive\\Documents\\Python\\pythonProject\\DataScience\\Data_Sets\\iowa_housing_train.csv"

#Load data
home_data = pd.read_csv(iowa_data_path)

#Selecting The Prediction Target
y = home_data.SalePrice

# Choosing "Features"
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Print 5 first rows of Features
print(X.head())


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)

#Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

#Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

#Print the top few validation predictions
print("top few validation predictions: ",iowa_model.predict((val_X).head()))

#Print the top few actual prices from validation data
print("top actual prices:\n",val_y.head())

#Print mean absolute error
val_mae = mean_absolute_error(val_y, val_predictions)
print("mean_absolute_error :",val_mae)

#Compare MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# A loop to find the ideal tree size from candidate_max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
mae_dict = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    mae_dict[max_leaf_nodes] = my_mae
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = min(mae_dict, key=mae_dict.get)
print("The best tree size is :",best_tree_size)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)