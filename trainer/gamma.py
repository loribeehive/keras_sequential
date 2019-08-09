
import numpy as np
import keras
import pickle

train_data = TRAIN_DATA
label_data = LABEL_DATA
test_data = TEST_DATA
test_label= TEST_LABEL

evaluateAdamAlpha
import alpha


class Gamma:
    def __init__(self, model, SEQUENCE_INDEX,history_All,Accuracy,JOBDIR):
         self.model = model
         self.sequence_id = SEQUENCE_INDEX
         self.VERBOSE = 0
         self.history_All= history_All
         self.BATCH_SIZE = 40000
         self.Accuracy = Accuracy
         self.gamma_floor = 0.3
         self.percentage_start = 0.1
         self.weights_path = JOBDIR + '/weights/'

    def find_gamma(self):
#################### train data input generator
        pred = self.model.predict(train_data ,batch_size=8000 ,verbose=1)
#################### multiclass prediction
        pred_true =pred[label_data==1]
        pred_false =-1 *pred[label_data==0]
        pred_loss =np.concatenate([pred_true ,pred_false])

        # plt.hist(pred[pred<0],bins=100,density=1)
        ACCCEIL =( 1 -self.Accuracy -0.001 ) *100

        percentile_threshold =np.arange(self.percentage_start, ACCCEIL ,0.5)
        #######cut off percentage exceed accuracy#####
        if percentile_threshold.size==0:
            print('no more data left, gamma tuning finish')
            return 0 ,0


        threshold_loss =[]
        gamma =[]
        for percent in percentile_threshold:
            loss =np.percentile(pred_loss, percent)
            threshold_loss.append(loss)
            gamma.append(np.log(1/percent-1) /(- 1 *loss))

        gamma_index =np.argmax(np.array(gamma ) >self.gamma_floor)
        #######or no more good gamma#####
        if gamma_index==0:
            print('next gamma not big enough, gamma tuning finish')
            return 0 ,0

        print('gamma trying is: ' +str(gamma[gamma_index] ) +'\n '+
              'loss threshold is: ' +str(threshold_loss[gamma_index] ) +'\n '+
              'cut off percentage is: ' +str(percentile_threshold[gamma_index] ) +'\n')

        gamma_chosen =gamma[gamma_index]
        if gamma_chosen >19.5:
            print('gamma too big,gamma tuning finish')
            return  0 ,0

        cut_off_percentage =percentile_threshold[gamma_index]
        return gamma_chosen ,cut_off_percentage


    # @title Gamma Trials
    def gamma_tune(self):
        print('\n GAMMA TUNING....\n')
        trialNum = 1
        accBest = 0
        gammaTry = []
        ALPHA = 0.001
        while True:
            if trialNum == 1:
                gamma_floor = 2
                percentage_start = 0.1
            else:
                gamma_floor = GAMMA * 1.5
                percentage_start = percent

            Accuracy = self.history_All[-1].history['acc'][-1]
            gamma, percent = self.find_gamma(Accuracy, gamma_floor, percentage_start)
            if gamma != 0:
                GAMMA = gamma

                #################### multi class loss fuunction
                def customLoss(yTrue, yPred):
                    relu4 = keras.activations.relu(yPred, alpha=0.0, max_value=None, threshold=-4.1)
                    sigmoid5 = 1 / (1 + K.exp(-1 * GAMMA * (relu4)))

                    return -(yTrue * sigmoid5 + (1 - yTrue) * (1 - sigmoid5))

                LOSS_FUNCTION = customLoss

                print('gamma tuning on alpha=' + str(ALPHA) + '\n')
                new_model = self.model.compile(loss=LOSS_FUNCTION, optimizer=keras.optimizers.Adam(lr=ALPHA), metrics=['accuracy'])
                history = new_model.fit(
                    train_data, label_data,
                    epochs=20, verbose=self.VERBOSE, batch_size=self.BATCH_SIZE, validation_data=[test_data, test_label],
                )

                acc = history.history['val_acc'][-1]
                if acc > accBest:
                    accBest = acc
                    trialNum = trialNum + 1
                    gammaTry.extend([trialNum, GAMMA, acc])
                    if self.SEQUENCE_INDEX == 1:
                        with open(self.weights_path + str(self.SEQUENCE_INDEX) + 'hrs_gamma_weights', 'wb') as fp:
                            pickle.dump(new_model.get_weights(), fp)
                    else:
                        weights = new_model.get_weights()
                        weights_0 = [np.concatenate((weights[0], weights[2]), axis=1),
                                     np.concatenate((weights[1], weights[3])),
                                     np.concatenate((weights[4], weights[6]), axis=1),
                                     np.concatenate((weights[5], weights[7])),
                                     np.concatenate((weights[8], weights[10]), axis=0),
                                     weights[9] + weights[11]]
                        with open(self.weights_path + str(self.SEQUENCE_INDEX) + 'hrs_gamma_weights', 'wb') as fp:
                            pickle.dump(weights_0, fp)
                    self.history_All.append(history)
                    # ALPHA = alpha.evaluateAdamAlpha(20 * 40, alpha.evaluateAdamAlpha)

                else:
                    print('valuation accuracy not improving!')
                    break

            else:
                break
        return gammaTry,self.history_All