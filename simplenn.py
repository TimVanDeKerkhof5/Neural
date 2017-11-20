from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime as dt
from fancyimpute import KNN
from sklearn import preprocessing as skp
from sklearn.model_selection import train_test_split

import os

#headers of categorical & continuous variables:
selectedcat = ['Geslacht','DiagnoseCode','DBC_Specialisme','vrgeschiedenis_myochardinfarct','vrgeschiedenis_PCI','vrgeschiedenis_CABG','vrgeschiedenis_CVA_TIA','vrgeschiedenis_vaatlijden','vrgeschiedenis_hartfalen','vrgeschiedenis_maligniteit','vrgeschiedenis_COPD','vrgeschiedenis_atriumfibrilleren','TIA','CVA_Niet_Bloedig','CVA_Bloedig','LV_Functie','dialyse','riscf_roken','riscf_familieanamnese','riscf_hypertensie','riscf_hypercholesterolemie','riscf_diabetes','roken','ECG_Ritme','Radialis','Femoralis','Brachialis','vd_1','vd_2','vd_3','graftdysfunctie']
selectedcon = ['Geboortedatum', 'lengte','gewicht','bloeddruk','HB','HT','INR','Glucose','Kreat','Trombocyten','Leukocyten','Cholesterol_totaal','Cholesterol_ldl']
selectedtarget = 'lbl'
#creation of the 'raw' dataframe:
nndf = pd.read_csv('nninput.csv',header=0,low_memory=False,encoding='ISO-8859-1')

def initialize():
	#initialization of dataframe used in logic.
	catdf = pd.DataFrame()

	#one-hot encoding all variables
	for feature in selectedcat:
		nndf[feature] = nndf[feature].fillna('N')
		dummies = pd.get_dummies(nndf[feature],prefix=feature)
		catdf[dummies.columns] = dummies.astype(float)

	#taking all continuous columns, purely for headers, then imputating with fancyimpute's K nearest neighbours
	precondf = pd.DataFrame(nndf[selectedcon])
	condf = pd.DataFrame(KNN(3).complete(nndf[selectedcon]))
	condf.columns = precondf.columns
	condf.index = precondf.index
	for feature in condf:
		condf[feature] = normalizedata(condf[feature])

	#add label (coded to int, not float)
	lblarr = []
	for item in nndf[selectedtarget]:
		if item:
			lblarr.append(1)
		else:
			lblarr.append(0)

	dflbl = pd.DataFrame()

	dflbl['lbl'] = lblarr
	dflbl['lbl'] = dflbl.astype(int)
	inputdf = pd.concat([catdf,condf,dflbl],axis=1,ignore_index=True)

	#unhash for csv file with the final input for the neural network:
	dftest, dftrain = train_test_split(inputdf, test_size=0.7)



def normalizedata(df):
	#normalization function that returns a normalized column.
	x = df.values.astype(float)
	x = pd.DataFrame(x)
	mms = skp.MinMaxScaler()
	x_s = mms.fit_transform(x)
	df = pd.DataFrame(x_s)
	return df

def dnn():
	training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename='train.csv',target_dtype=np.int,features_dtype=np.float32)
	test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename='test.csv',target_dtype=np.int,features_dtype=np.float32)

	feature_columns = [tf.feature_column.numeric_column("x",shape=[99])]

	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[20,40,20],model_dir="/tmp/read_model")

	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.array(training_set.data)},y=np.array(training_set.target),num_epochs=None,shuffle=True)

	classifier.train(input_fn=train_input_fn,steps=2000)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.array(test_set.data)},y=np.array(test_set.target),num_epochs=1,shuffle=False)

	accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



def main():
	#initialize()
	dnn()

main()

