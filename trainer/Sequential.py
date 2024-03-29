
from __future__ import absolute_import, division, print_function

import pathlib
import multiprocessing


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
import pickle
import trainer.model as initial_model
import numpy as np
from numpy import median
from tensorflow.keras.layers import *
from keras.layers import *

#######first topology###############

import numpy as np
import keras
import pickle


class Sequential:
    def __init__(self, weights,CONCAT_UNIT_SIZE,INPUT_SHAPE,initializer):
         self.old_weights = weights
         self.CONCAT_UNIT_SIZE = CONCAT_UNIT_SIZE
         self.INPUT_SHAPE =INPUT_SHAPE
         self.initializer = initializer


    ##############################
    def pad_weights(self, weights,new_shape):
        padded_weights = np.zeros([new_shape,weights.shape[1]])
        padded_weights[-weights.shape[0]:,:weights.shape[1]] = weights
        return padded_weights

    def concaternate_padded(self):
        #########input Layer#########
        padded_weights=[self.pad_weights(self.old_weights[0],self.INPUT_SHAPE),
                    self.old_weights[1]]
        #########pad zeros into hidden layer#########
        for i in range(len(self.old_weights)-4):
            ind = i+2
            if ind==2:
                if (ind % 2) == 0:
                    padded_weights.append(
                        self.pad_weights(self.old_weights[ind], self.old_weights[ind].shape[0] + self.CONCAT_UNIT_SIZE))
                else:
                    padded_weights.append(self.old_weights[ind])
            else:
                if (ind % 2) == 0:
                    padded_weights.append(self.pad_weights(self.old_weights[ind],self.old_weights[ind].shape[0]*2))
                else:
                    padded_weights.append(self.old_weights[ind])
        #########output Layer#########
        padded_weights.append(self.old_weights[-2])
        padded_weights.append(self.old_weights[-1])
        return padded_weights


    def build_sequential_model(self):
        padded_weights=self.concaternate_padded()
        #########input Layer#########
        input0 = Input(shape=([self.INPUT_SHAPE]))

        dense1 = Dense(padded_weights[1].shape[0],  name='input_a',
                       weights=[padded_weights[0], padded_weights[1]], trainable=False,activation='relu')(input0)
        dense2 = Dense(self.CONCAT_UNIT_SIZE, kernel_initializer=self.initializer,
                       bias_initializer=self.initializer, name='input_b',activation='relu')(input0)
        # dense2 = Dense(self.CONCAT_UNIT_SIZE, kernel_initializer='zeros',
        #                bias_initializer='zeros', activation='relu', name='input_b')(input0)

        merged1 = concatenate([dense2, dense1])
        # merged1=keras.activations.relu(merged1)
        #########hidden layer#########
        hidden_layer_N =int(len(padded_weights)/2)-2
        hidden_dense = []
        merged = []
        for i in range(hidden_layer_N):
            if i == 0:
                #########first hidden layer connects to input#########
                merged.append(merged1)

            hidden_dense.append(Dense(padded_weights[2*(i+1)+1].shape[0], name='hidden_layer_'+str(i)+'_a',activation='relu',
                       weights=[padded_weights[2*(i+1)], padded_weights[2*(i+1)+1]], trainable=False)(merged[-1]))
            hidden_dense.append(Dense(padded_weights[2*(i+1)+1].shape[0], kernel_initializer=self.initializer,
                       bias_initializer=self.initializer, activation='relu',name='hidden_layer_'+str(i)+'_b')(merged[-1]))
            # hidden_dense.append(Dense(padded_weights[2 * (i + 1) + 1].shape[0], kernel_initializer='zeros',
            #            bias_initializer='zeros', activation='relu', name='hidden_layer_'+str(i)+'_b')(merged[-1]))

            if i!=hidden_layer_N-1:

                merged.append(concatenate([hidden_dense[-1], hidden_dense[-2]]))
        #########output Layer#########
        output1 = Dense(padded_weights[-1].shape[0], name='output_a', activation='softmax',weights=[padded_weights[-2], padded_weights[-1]],
                        trainable=False)(hidden_dense[-2])
        output2 = Dense(padded_weights[-1].shape[0], kernel_initializer=self.initializer,
                        bias_initializer=self.initializer, activation='softmax',name='output_b')(hidden_dense[-1])
        # output2 = Dense(padded_weights[-1].shape[0], activation='softmax', kernel_initializer='zeros',
        #                 bias_initializer='zeros', name='output_b')(hidden_dense[-1])

        out = add([output1, output2])
        # out = keras.activations.softmax(out_temp)
        model = keras.models.Model(inputs=input0, outputs=out)
        # self.compile_model(model)
        return model

    # def compile_model(self, model):
    #
    #     return model
    # def compile_model(self, model):
    #     model.compile(
    #         loss='categorical_crossentropy',
    #         optimizer=keras.optimizers.Adam(lr=self.learning_rate),
    #         metrics=['accuracy'])
    #     return model