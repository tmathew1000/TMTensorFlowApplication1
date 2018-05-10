import pandas as pd
census=pd.read_csv("census_data.csv")
print(census.head())
census['income'].unique()
def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1
census['income']= census['income'].apply(label_fix)

#Perform Train Test Split on census data

from sklearn.model_selection import train_test_split
x_data=census.drop('income', axis=1)
y_labels=census['income']
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=101)

#Tensorflow offers an API called estimator
#model=builtin_function_or_method.estimator.LinearClassifier(feature_columns=feat_cols)
#model.train(input_fn=input_func,steps=5000)
#prepare feature columns and input function

print(census.columns)

import tensorflow as tf

#Continous or Categorical Column Types

#create the tf.feature_columns for the categorical values. 
#use vocabulary lists or just use hash buckets
#age	workclass	fnlwgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country	income
workclass =tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
education =tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
marital_status =tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
occupation =tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
relationship =tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
race =tf.feature_column.categorical_column_with_hash_bucket("race", hash_bucket_size=1000)
sex =tf.feature_column.categorical_column_with_vocabulary_list("sex", ["Male", "Female"])
native_country =tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#Create the continuous feature Column s for the continous values using numeric_column
age=tf.feature_column.numeric_column("age")
fnlwgt=tf.feature_column.numeric_column("fnlwgt")
education_num=tf.feature_column.numeric_column("education_num")
capital_gain=tf.feature_column.numeric_column("capital_gain")
capital_loss=tf.feature_column.numeric_column("capital_loss")
hours_per_week=tf.feature_column.numeric_column("hours_per_week")

feat_cols = [age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]

#Create input Function
#Batch size is up to you. How many records you want to read at one time.
#num_epochs = how many times shoudl the data be passed through the model. if you have 1000 records, and pass through training 3 times, you should have epochs as 3
#if there are 1000 records and batch size is 100, then there is 10 batches to get 1 epoch to be completed.


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

#Create your model with tf.estimator
#Create a Linear Classifier
model=tf.estimator.LinearClassifier(feature_columns=feat_cols)


#Train your model on data for atleast 5000 steps. How many iterations this training will be done. 
model.train(input_fn=input_func, steps =5000)

#Evaluation
#Create a prediction input function. Provide shuffle=false
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

#use model.predict() to pass in your input fucntion. this will prodiuce a generator or predictions, which you can then transform into a list with list()
predictions = list(model.predict(input_fn=pred_fn))


#prediction[0]

#Create a list of only the class_ids key values from the prediction list of dictionaries, these are the predictions you will use to compare against the real y_test values

final_preds=[]
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

#final_preds[:10]

#Import Classification_report from sklean.metrics and see if you can figure out how to use it to easioly get a full report of your model's performance on the test data
from sklearn.metrics import classification_report

print(classification_report(y_test, final_preds))
