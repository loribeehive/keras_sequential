
# ==============================================================================
"""Implements the Keras Sequential model."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements the Keras Sequential model."""

from builtins import range

import keras
import pathlib
import keras.backend.tensorflow_backend as K
from keras import layers
from keras import models
from keras.backend import relu
import argparse
import pandas as pd
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

INTERESTING_CLASS_ID = 0 # Choose the class of interest

import numpy as np
bins = np.array([50,100,150,200,250,500,1100])
CSV_COLUMNS = ('time','IOPS','Throughput','MBps')

one_hour = 12





def model_fn(input_dim,
             labels_dim,
             hidden_units,
             learning_rate=0.1):

  K.set_learning_phase(False)
  model = models.Sequential()
  hidden_units = [int(units) for units in hidden_units.split(',')]

  for units in hidden_units:
      model.add(layers.Dense(units, activation=relu, input_shape=[input_dim],
                            kernel_initializer='glorot_uniform',
                            ))
      # model.add(layers.Dropout(0.5))
      # model.add(layers.BatchNormalization(epsilon=1e-03, momentum=0.9, weights=None))
      input_dim = units
      #                 activity_regularizer=tf.keras.regularizers.l1(0.01)


  # Add a dense final layer with sigmoid function.
  model.add(layers.Dense(labels_dim, activation='softmax'))
  # compile_model(model, learning_rate)
  return model


def weighted_loss(assign_w):


    def customLoss(yTrue, yPred):

             weights = [assign_w, 1, 1, 1, 1, 1, 1, 1]
             target = yTrue

             # output = yPred
             yPred /= tf.reduce_sum(yPred,
                                reduction_indices=len(yPred.get_shape()) - 1,
                                keep_dims=True)
                # manual computation of crossentropy
             epsilon = K._to_tensor(tf.keras.backend.epsilon(), yPred.dtype.base_dtype)
             yPred = tf.clip_by_value(yPred, epsilon, 1. - epsilon)
             yPred = tf.log(yPred)

             ######apply weights here###############
             mask = K.cast(K.expand_dims(weights, axis=-1), dtype='float32')
             tensor_shape = yPred.get_shape()
             # x = tf.add(x, tf.constant(1, shape=x.shape))
             yPred_stack = []
             for i in range(tensor_shape[1]):
                 mask_i = K.cast(K.expand_dims(mask[i], axis=-1), dtype='float32')
                 yPred_i = K.cast(K.expand_dims(yPred[:, i], axis=-1), dtype='float32')
                 yPred_stack.append(K.dot(yPred_i, mask_i))

             output = tf.reshape(tf.stack(yPred_stack, axis=1, name='stack'), [-1, tensor_shape[1]])

             return - tf.reduce_sum(target * output,
                                       reduction_indices=len(output.get_shape()) - 1)

    return customLoss
    # return - tf.reduce_sum(yTrue * tf.log(yPred))


def first_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_true, INTERESTING_CLASS_ID), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_preds, class_id_true), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc

def other_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    class_type_mask = K.cast(K.greater(class_id_true, INTERESTING_CLASS_ID), 'int32')
    class_acc_tensor = K.cast(K.greater(class_id_preds, INTERESTING_CLASS_ID), 'int32') * class_type_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(class_type_mask), 1)
    return class_acc
#

# assign_w=0.03

# def compile_model(model, learning_rate):
#   model.compile(
#       loss=weighted_loss(assign_w),
#       optimizer=keras.optimizers.Adam(lr=learning_rate),
#       metrics=[first_class_accuracy,other_class_accuracy,'accuracy'])
#   return model


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)
  signature = predict_signature_def(
      inputs={'MBps': model.inputs[0]}, outputs={'Category': model.outputs[0]}
  )
  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
    )
    builder.save()


def generator_input(filenames, training_history, batch_size):
  """Produce features and labels needed by keras fit_generator."""
  # if tf.gfile.IsDirectory(filenames):
  #     files = tf.gfile.ListDirectory(filenames)
  # else:
  #     files = filenames

  while True:

      files = tf.gfile.Glob(filenames)
      for file in files:
          file = str(file)
          data = np.load(file)


          input=data['input']
          input = (input-5.036841015168413)/12.866818115879605
          label = data['label']
          idx_len = input.shape[0]
          for index in range(0, idx_len, batch_size):
            yield (input[index:min(idx_len, index + batch_size)],
                   label[index:min(idx_len, index + batch_size)])

def generator_input_once(filenames, training_history):



      files = tf.gfile.Glob(filenames)
      for file in files:
          file = str(file)
          data = np.load(file)


          input=data['input']
          input = (input-5.036841015168413)/12.866818115879605

          label = data['label']
          return input,label
          # idx_len = input.shape[0]
          # for index in range(0, idx_len, batch_size):
          #   yield (input[index:min(idx_len, index + batch_size)],
          #          label[index:min(idx_len, index + batch_size)])



# def process_data(data,training_history):
#
#
#     data_read = data[['time', 'MBps']]
#     # insert label
#     data_read['Bucket_Index'] = np.digitize(data_read['MBps'], bins, right=False)
#     # make time series as TimeIndex
#     idx = pd.to_datetime(data_read['time'])
#     data_read = data_read.drop(['time'], axis=1)
#     data_read = data_read.set_index(idx)
#
#     ####take the mean of every five mins
#     data_sample = data_read['MBps'].resample('5Min').mean()
#     data_sample = data_sample.fillna(0)
#     # resample label to max(1H)
#     label_data = data_read['Bucket_Index'].resample('1H').max()
#     label_data  = label_data .fillna(0)
#
#     if data_sample.empty or label_data.empty:
#         return (np.array([]),np.array([]))
#
#     disk_trainX,disk_trainY = reshape_input(data_sample, label_data,
#                                    training_history)
#
#
#     return (disk_trainX,disk_trainY)
#
#
#
# def reshape_input(data, label,training_history):
#     data_SIZE = len(label.values)-1
#     subLabel = np.zeros((data_SIZE, len(bins)+1))
#     label=label.iloc[:-1]
#     subLabel[np.arange(data_SIZE), np.array(label.values).astype(int)] = 1
#     subLabel = subLabel[training_history:, :]
#
#     LENN = subLabel.shape[0]
#     subData = np.array(data.values)
#     subData = subData[0:one_hour * (LENN + (training_history - 1))]
#     s_Data=np.array([])
#     for j in range(LENN):
#
#         s_data = subData[j * one_hour:(j * one_hour + one_hour * training_history)]
#         if j == 0:
#             s_Data = s_data
#             #####if only one row of data in total
#             s_Data = s_Data.reshape(-1, one_hour * training_history)
#
#         else:
#             s_Data = np.vstack((s_Data, s_data))
#
#
#     return (s_Data, subLabel)






#
# def compile_model(model, learning_rate):
#   model.compile(
#       loss='categorical_crossentropy',
#       optimizer=keras.optimizers.Adam(lr=learning_rate),
#       metrics=['accuracy'])
#   return model
#
#
# def to_savedmodel(model, export_path):
#   """Convert the Keras HDF5 model into TensorFlow SavedModel."""
#
#   builder = saved_model_builder.SavedModelBuilder(export_path)
#   signature = predict_signature_def(
#       inputs={'MBps': model.inputs[0]}, outputs={'Category': model.outputs[0]}
#   )
#   with K.get_session() as sess:
#     builder.add_meta_graph_and_variables(
#         sess=sess,
#         tags=[tag_constants.SERVING],
#         signature_def_map={
#             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
#         }
#     )
#     builder.save()














# def eval_weighted_loss(y_true, y_pred):
#     target = y_true
#     ######output times weights here###############
#     mask = K.cast(K.expand_dims(weights, axis=-1), dtype='float32')
#     tensor_shape = y_pred.get_shape()
#     # x = tf.add(x, tf.constant(1, shape=x.shape))
#     yPred_stack = []
#     for i in range(tensor_shape[1]):
#         mask_i = K.cast(K.expand_dims(mask[i], axis=-1), dtype='float32')
#         yPred_i = K.cast(K.expand_dims(y_pred[:, i], axis=-1), dtype='float32')
#         yPred_stack.append(K.dot(yPred_i, mask_i))
#     output = tf.reshape(tf.stack(yPred_stack, axis=1, name='stack'), [-1, tensor_shape[1]])
#
#     return output[0, 7]
#
# def eval_loss(y_true, y_pred):
#     target = y_true
#     ######output times weights here###############
#     mask = K.cast(K.expand_dims(weights, axis=-1), dtype='float32')
#     tensor_shape = y_pred.get_shape()
#     # x = tf.add(x, tf.constant(1, shape=x.shape))
#     yPred_stack = []
#     for i in range(tensor_shape[1]):
#         mask_i = K.cast(K.expand_dims(mask[i], axis=-1), dtype='float32')
#         yPred_i = K.cast(K.expand_dims(y_pred[:, i], axis=-1), dtype='float32')
#         yPred_stack.append(K.dot(yPred_i, mask_i))
#     output = tf.reshape(tf.stack(yPred_stack, axis=1, name='stack'), [-1, tensor_shape[1]])
#
#     return y_pred[0, 7]
#