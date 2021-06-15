# https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pickle
from whylogs import get_or_create_session
import os


# Set whylogs variables
os.environ["WHYLABS_API_KEY"] = "<WHYLABS_API_KEY>"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "<WHYLABS_ORG_ID>"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "<WHYLABS_DATASET_ID>"
# Whylogs session
wl_session = get_or_create_session()

# Reading our data
df = pd.read_csv("model/heart.csv")

y = df.target.values
x_data = df.drop(['target'], axis = 1)

wl_session.log_dataframe(x_data, 'training.input.raw')

# Normalize
x = normalize(x_data)
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
wl_session.log_dataframe(pd.DataFrame.from_records(x_train), 'training.input.normalized')
wl_session.log_dataframe(pd.DataFrame.from_records(x_test), 'test.input.normalized')


# Random Forest Classification
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)

acc = rf.score(x_test, y_test)*100
print("Accuracy Score of our model: {:.2f}%".format(acc))

# Save the built model to our dployment folder
with open('deployment_folder/model.pkl', 'wb') as f:
    pickle.dump(rf, f)

