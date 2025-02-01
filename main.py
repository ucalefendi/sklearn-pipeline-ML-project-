import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error



data = pd.read_csv("vehicle_emission.csv")

print(data.head())
data.info()



###############################################
#create features and target variable
X = data.drop(["CO2_Emissions"],axis=1)
y = data["CO2_Emissions"]


#split categorical and numerical features
numerical_cols = ['Model_Year','Engine_Size','Cylinders','Fuel_Consumption_in_City(L/100 km)','Fuel_Consumption_in_City_Hwy(L/100 km)','Fuel_Consumption_comb(L/100km)','Smog_Level']
categorical_cols = ['Make','Model','Vehicle_Class','Transmission']


numerical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


categorical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="most_frequent")),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#join pipelines together
preprocessor = ColumnTransformer([
    ('num',numerical_pipeline,numerical_cols),
    ('cat',categorical_pipeline,categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestRegressor())
])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pipeline.fit(X_train,y_train)

prediction = pipeline.predict(X_test)

#view the encoding that done
encoded_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
# print('>>>>>',encoded_cols)

#evaluate model accuracy

mse = mean_squared_error(y_test,prediction)
rmse = np.sqrt(mse)


r2 = r2_score(y_test,prediction)

mae = mean_absolute_error(y_test,prediction)


print(f"Model performance: ")
print(f"R2 Score: {r2} ")
print(f"Root Mean Square error: {rmse}")
print(f"Mean Absolute Error: {mae}")


#Outputs-->> in below
# Model performance: 
# R2 Score: 0.975383800272614 
# Root Mean Square error: 9.957888282032146
# Mean Absolute Error: 3.1328342245989305


joblib.dump(pipeline,'vehicle_emission_pipeline.joblib')

