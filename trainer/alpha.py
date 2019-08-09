# @title compute alpha
import numpy as np
from tensorflow.keras import backend as K
import keras
####model need to recompile with new learning rate!!!!



train_data = TRAIN_DATA
label_data = LABEL_DATA
test_data = TEST_DATA
test_label= TEST_LABEL
class Alpha:
    def __init__(self, model, SEQUENCE_INDEX,history_All,Accuracy,JOBDIR):
         self.model = model
         self.SEQ_ID = SEQUENCE_INDEX
         self.VERBOSE = 0
         self.history_All= history_All
         self.BATCH_SIZE = 40000
         self.Accuracy = Accuracy
         self.ALPHA_PERC = 0.2
         self.CONCAT_UNIT_SIZE = 300
         self.FIRST_MODEL_FIRST_LAYER = 200

    def computeAlpha(self,x, w, b, w_grad, b_grad):
        Epsilon = 1e-5
        alpha0 = np.dot(x, w) + b
        alpha1 = np.dot(x, w_grad) + b_grad
        alpha = np.divide(alpha0, alpha1, out=np.zeros_like(alpha0), where=alpha1 != 0)
        return alpha0, alpha


    def pad_grad(self,grads, new_shape):
        padded_grads = np.zeros([grads.shape[0], new_shape])
        padded_grads[:, :grads.shape[1]] = grads
        return padded_grads


    def evaluateAdamAlpha(self,EPOCH, CUT_OFF_PERCENT):
        print('\n CHANGING ALPHA')
        weights = self.model.get_weights()
        grads = self.model.optimizer.weights
        if len(weights) > 6:
            #       pad_shape=int(300*np.log2(K.eval(grads[1]).shape[0]/36))+100
            pad_shape = self.CONCAT_UNIT_SIZE * self.SEQ_ID + self.FIRST_MODEL_FIRST_LAYER
            w1 = np.concatenate((weights[0], weights[2]), axis=1)
            b1 = np.concatenate((weights[1], weights[3]))
            m1_grad = self.pad_grad(K.eval(grads[1]), pad_shape)
            v1_grad = self.pad_grad(K.eval(grads[7]), pad_shape)

            bm1_grad = np.pad(K.eval(grads[2]), (0, pad_shape - K.eval(grads[2]).shape[0]), 'constant', constant_values=0)
            bv1_grad = np.pad(K.eval(grads[8]), (0, pad_shape - K.eval(grads[8]).shape[0]), 'constant', constant_values=0)
        else:
            w1 = weights[0]
            b1 = weights[1]
            m1_grad = K.eval(grads[1])
            v1_grad = K.eval(grads[7])
            bm1_grad = K.eval(grads[2])
            bv1_grad = K.eval(grads[8])

        #############adam gradients############
        w1_grad = np.nan_to_num((m1_grad / (1 - 0.9 ** (EPOCH))) / np.nan_to_num(np.sqrt(v1_grad / (1 - 0.999 ** (EPOCH)))))
        b1_grad = np.nan_to_num(
            (bm1_grad / (1 - 0.9 ** (EPOCH))) / np.nan_to_num(np.sqrt(bv1_grad / (1 - 0.999 ** (EPOCH)))))

        x_1, alpha1 = self.computeAlpha(train_data, w1, b1, w1_grad, b1_grad)
        a = np.sort(alpha1, axis=1)
        b = np.argmax(a > 0, axis=1)
        #############get every data point alpha##############
        alphaA = a[np.arange(0, b.shape[0], 1), b]

        #############pick an alpha based on percentage##############
        alphaFind = np.percentile(alphaA, CUT_OFF_PERCENT)
        print('\nalpha at ' + str(CUT_OFF_PERCENT) + '% is:' + str(alphaFind) + '\n')
        #############forgot whats this###
        #     lr_tt=alphaFind*0.1/np.sqrt(0.001)
        #     print('final tune alpha is :'+str(lr_tt))
        return alphaFind

    #@title changing LR
    def alpha_trial(self,num,epoch_last,alpha_epoch):
         for i in range(num):
              ALPHA=self.evaluateAdamAlpha(epoch_last*40,self.ALPHA_PERC)
              new_model = self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=ALPHA), metrics=['accuracy'])
              epoch_last=alpha_epoch
              history=new_model.fit(
               train_data,label_data,
               epochs=alpha_epoch,verbose=self.VERBOSE,batch_size=self.BATCH_SIZE,validation_data=[test_data,test_label],
              )
              self.history_All.append(history)
