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
selectedcat = ['Geslacht','DiagnoseCode','DBC_Specialisme','vrgeschiedenis_myochardinfarct','vrgeschiedenis_PCI','vrgeschiedenis_CABG','vrgeschiedenis_CVA_TIA','vrgeschiedenis_vaatlijden','vrgeschiedenis_hartfalen','vrgeschiedenis_maligniteit','vrgeschiedenis_COPD','vrgeschiedenis_atriumfibrilleren','TIA','CVA_Niet_Bloedig','CVA_Bloedig','LV_Functie','dialyse','riscf_roken','riscf_familieanamnese','riscf_hypertensie','riscf_hypercholesterolemie','riscf_diabetes','roken','ECG_Ritme','Radialis','Femoralis','Brachialis','vd_1','vd_2','vd_3']
selectedcon = ['Geboortedatum', 'lengte','gewicht','bloeddruk','HB','HT','INR','Glucose','Kreat','Trombocyten','Leukocyten','Cholesterol_totaal','Cholesterol_ldl']
selectedtarget = 'lbl'
#creation of the 'raw' dataframe:
nndf = pd.read_csv('nninput.csv',header=0,low_memory=False,encoding='ISO-8859-1')

#dfs and arrays for the CNN (quick and dirty solution)
cnnlbltrain = []
cnnlbltest = []
testcnn = pd.DataFrame()
traincnn = pd.DataFrame()


def initforcnn():
	#initialization of dataframe used in logic.
	catdf = pd.DataFrame()

        #one-hot encoding all variables
	for feature in selectedcat:
		nndf[feature] = nndf[feature].fillna('N')
		dummies = pd.get_dummies(nndf[feature],prefix=feature)
		catdf[dummies.columns] = dummies.astype(np.float32)
        #taking all continuous columns, purely for headers, then imputating with fancyimpute's K nearest neighbours
	precondf = pd.DataFrame(nndf[selectedcon])
	condf = pd.DataFrame(KNN(3).complete(nndf[selectedcon]))
	condf.columns = precondf.columns
	condf.index = precondf.index
	for feature in condf:
		condf[feature] = normalizedata(condf[feature])
		condf[feature] = condf[feature].astype(np.float32)

        #add label (coded to int, not float)
	lblarr = []
	for item in nndf[selectedtarget]:
		if item:
			lblarr.append(0)
		else:
			lblarr.append(1)

	dflbl = pd.DataFrame()

	dflbl['lbl'] = lblarr
	dflbl['lbl'] = dflbl.astype(int)
	inputdf = pd.concat([catdf,condf,dflbl],axis=1,ignore_index=True)
        #unhash for csv file with the final input for the neural network:
	dftest, dftrain = train_test_split(inputdf, test_size=0.85)

        #set parameters for the CNN
	cnntrain = dftrain[dftrain.columns.difference([97])]
	cnntest = dftest[dftest.columns.difference([97])]
	cnnlbltrain = dftrain[97].values
	cnnlbltest = dftest[97].values
	return cnntrain, cnntest, cnnlbltrain, cnnlbltest


def normalizedata(df):
	#normalization function that returns a normalized column.
	x = df.values.astype(float)
	x = pd.DataFrame(x)
	mms = skp.MinMaxScaler()
	x_s = mms.fit_transform(x)
	df = pd.DataFrame(x_s)
	return df

def cnn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 97, 1, 1])
# Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 97, 1, 1]
  # Output Tensor Shape: [batch_size, 91, 1, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 1],
		padding="same",
		activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x1 filter and stride of 2
  # Input Tensor Shape: [batch_size, 97, 1, 32]
  # Output Tensor Shape: [batch_size, 96/2, 1, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 97/2, 1, 32]
  # Output Tensor Shape: [batch_size, 97/2, 1, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 1],
		padding="same",
		activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x1 filter and stride of 2
  # Input Tensor Shape: [batch_size, 97/2, 1, 64]
  # Output Tensor Shape: [batch_size, 97/4, 1, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 97/4, 1, 64]
  # Output Tensor Shape: [batch_size, 97/4 * 1 * 64]
	pool2_flat = tf.reshape(pool2, [- 1, 24 * 1 * 64])

  # Dense Layer
  # Densely connected layer with 2000 neurons
  # Input Tensor Shape: [batch_size, 97/4 * 1 * 64]
  # Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=2000, activation=tf.nn.relu)
  # Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 2000]
  # Output Tensor Shape: [batch_size, 2]
	logits = tf.layers.dense(inputs=dropout, units=2)

	predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
	#initialize()
	#dnn()
	trainsetcnn, testsetcnn, trainlblcnn, testlblcnn = initforcnn()
	train_data = trainsetcnn.as_matrix()
	train_labels = np.asarray(trainlblcnn, dtype=np.int32)
	eval_data = testsetcnn.as_matrix()
	eval_labels = np.asarray(testlblcnn, dtype=np.int32)

	print(train_data)
	cnn_classifier = tf.estimator.Estimator(
		model_fn=cnn, model_dir="/tmp/cnnclassmodel")

	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	cnn_classifier.train(
		input_fn=train_input_fn,
		steps=100000,
		hooks=[logging_hook])

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)

	eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)
main()

