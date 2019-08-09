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
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle
from trainer.Sequential import Sequential
import trainer.model as initial_model

# INPUT_SIZE = 288
ONE_HOUR=12
bins = np.array([50,100,150,200,250,500,1100])
# CLASS_SIZE = len(bins)

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'





def train_and_evaluate(args):

  CLASS_SIZE = len(bins)+1

  # hidden_units = [int(units) for units in args.hidden_units.split(',')]

  try:
    os.makedirs(args.job_dir)
  except:
    pass

  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  checkpoint_path = CHECKPOINT_FILE_PATH
  if not args.job_dir.startswith('gs://'):
    checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

  # Model checkpoint callback.
  checkpoint = ModelCheckpoint(
      checkpoint_path,
      monitor='val_loss',
      verbose=1,
      period=args.checkpoint_epochs,
      mode='min')

  tb_log = TensorBoard(
      log_dir=os.path.join(args.job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  callbacks = [checkpoint,tb_log]
  sequential_train = [int(hour) for hour in args.sequential_train.split(',')]
  seq_id = 0
  sequential_models=[]
  for hours in sequential_train:
      history_all = []

      if seq_id==0:
          hidden_units = args.hidden_units
          learning_rate = args.learning_rate
          INPUT_DIM = hours * ONE_HOUR

          ###########fully connected model#############
          first_model = initial_model.model_fn(INPUT_DIM, CLASS_SIZE,
                                      hidden_units, learning_rate
                                      )
          train_file_names = args.train_files + str(hours) + 'hrs/train/*npz'
          eval_file_names = args.eval_files + str(hours) + 'hrs/eval/*npz'
          print("\n\ntraining "+str(hours)+'hrs!\n\n')
          history_all.append(first_model.fit_generator(
              initial_model.generator_input(train_file_names, hours,args.train_batch_size),
              validation_data=initial_model.generator_input(eval_file_names, hours,args.eval_batch_size),
              steps_per_epoch=args.train_steps,validation_steps = args.eval_steps,
              epochs=args.num_epochs,
              callbacks=callbacks))


          weights = first_model.get_weights()
          with open(os.path.join(args.job_dir, 'weights',str(hours) + 'hrs_weights'), 'wb') as fp:
                  pickle.dump(weights, fp)
          DISK_MODEL = 'disk_model.hdf5'
          if args.job_dir.startswith('gs://'):
            first_model.save(DISK_MODEL)
            copy_file_to_gcs(args.job_dir, DISK_MODEL)
          else:
            first_model.save(os.path.join(args.job_dir, DISK_MODEL))

          seq_id = seq_id + 1

      else:
          with open(os.path.join(args.job_dir, 'weights',str(sequential_train[seq_id - 1]) + 'hrs_weights'), 'rb') as fp:
              weights_0 = pickle.load(fp)
              ######sequential(weights, CONCAT_UNIT_SIZE, INPUT_SHAPE, learning_rate)
          seq = Sequential(weights_0,args.CONCAT_UNIT_SIZE,hours * ONE_HOUR,0.01)

          sequential_models.append(seq.build_sequential_model())


          ###########sequential model#############
          train_file_names = args.train_files+str(hours)+'hrs/train/*npz'
          eval_file_names = args.eval_files +str(hours) + 'hrs/eval/*npz'
          print("\n\ntraining " + str(hours) + 'hrs!\n\n')
          history_all.append(sequential_models[-1].fit_generator(
              initial_model.generator_input(train_file_names, hours, args.train_batch_size),
              validation_data=initial_model.generator_input(eval_file_names, hours, args.eval_batch_size),
              steps_per_epoch=args.train_steps, validation_steps=args.eval_steps,
              epochs=args.num_epochs,
              callbacks=callbacks))
          weights = sequential_models[-1].get_weights()
          weights_0=[]
          for i in range(int(len(weights)/4)):
              if i==int(len(weights)/4)-1:
                  weights_0.extend([np.concatenate((weights[i*4], weights[i*4+2]), axis=0),
                       (weights[i*4+1] + weights[i*4+3])])
              else:
                  weights_0.extend([np.concatenate((weights[i * 4], weights[i * 4 + 2]), axis=1),
                                    np.concatenate((weights[i * 4 + 1] , weights[i * 4 + 3]))])
          # weights_0 = [np.concatenate((weights[0], weights[2]), axis=1),
          #              np.concatenate((weights[1], weights[3])),
          #              np.concatenate((weights[4], weights[6]), axis=1),
          #              np.concatenate((weights[5], weights[7])),
          #              np.concatenate((weights[8], weights[10]), axis=0)
          #               weights[9] + weights[11]]

          with open(os.path.join(args.job_dir, 'weights',str(hours) + 'hrs_weights'), 'wb') as fp:
              pickle.dump(weights_0, fp)
          DISK_MODEL = 'disk_model' + str(hours) + '.hdf5'
          if args.job_dir.startswith('gs://'):
            sequential_models[-1].save(DISK_MODEL)
            copy_file_to_gcs(args.job_dir, DISK_MODEL)
          else:
            sequential_models[-1].save(os.path.join(args.job_dir, DISK_MODEL))
          seq_id = seq_id + 1

  with open(args.job_dir+ 'histroy_all', 'wb') as fp:
      pickle.dump(history_all, fp)


  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.



# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
      output_f.write(input_f.read())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-files',
      nargs='+',
      help='Training file local or GCS',
      default='/Volumes/TOSHIBA EXT/train_input/')
  parser.add_argument(
      '--eval-files',
      nargs='+',
      help='Evaluation file local or GCS',
      default='/Volumes/TOSHIBA EXT/train_input/')
  parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='/Users/xuerongwan/Documents/keras_job')
  parser.add_argument(
      '--sequential-train',
      default='3,6,12',
      help='number of hours of input')
  parser.add_argument(
      '--hidden_units',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
           '`64 32` means first layer has 64 nodes and second one has 32.',
      default='200,100,50',
      )
  parser.add_argument(
      '--train-steps',
      type=int,
      default=60,
      help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=10,
      type=int)
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=30000,
      help='Batch size for training steps')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=30000,
      help='Batch size for evaluation steps')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.001,
      help='Learning rate for SGD')
  parser.add_argument(
      '--eval-frequency',
      default=10,
      help='Perform one evaluation per n epochs')
  parser.add_argument(
      '--first-layer-size',
      type=int,
      default=256,
      help='Number of nodes in the first layer of DNN')
  parser.add_argument(
      '--CONCAT-UNIT-SIZE',
      type=int,
      default=100,
      help='Number of nodes in the first layer of DNN')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=2,
      help='Number of layers in DNN')
  parser.add_argument(
      '--scale-factor',
      type=float,
      default=0.25,
      help="""Rate of decay size of layer for Deep Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
  parser.add_argument(
      '--eval-num-epochs',
      type=int,
      default=1,
      help='Number of epochs during evaluation')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=50,
      help='Maximum number of epochs on which to train')
  parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=10,
      help='Checkpoint per n training epochs')

  args, _ = parser.parse_known_args()
  train_and_evaluate(args)


######################################google code call back########################################

# class ContinuousEval(Callback):
#   """Continuous eval callback to evaluate the checkpoint once
#
#      every so many epochs.
#   """
#
#   def __init__(self,
#                eval_frequency,
#                eval_files,
#                learning_rate,
#                job_dir,
#                steps=1000):
#     self.eval_files = eval_files
#     self.eval_frequency = eval_frequency
#     self.learning_rate = learning_rate
#     self.job_dir = job_dir
#     self.steps = steps
#
#   def on_epoch_begin(self, epoch, logs={}):
#     """Compile and save model."""
#     if epoch > 0 and epoch % self.eval_frequency == 0:
#       # Unhappy hack to work around h5py not being able to write to GCS.
#       # Force snapshots and saves to local filesystem, then copy them over to GCS.
#       model_path_glob = 'checkpoint.*'
#       if not self.job_dir.startswith('gs://'):
#         model_path_glob = os.path.join(self.job_dir, model_path_glob)
#       checkpoints = glob.glob(model_path_glob)
#       if len(checkpoints) > 0:
#         checkpoints.sort()
#         disk_model = load_model(checkpoints[-1])
#         disk_model = model.compile_model(disk_model, self.learning_rate)
#         loss, acc = disk_model.evaluate_generator(
#             model.generator_input(self.eval_files, chunk_size=CHUNK_SIZE),
#             steps=self.steps)
#         print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
#             epoch, loss, acc, disk_model.metrics_names))
#         if self.job_dir.startswith('gs://'):
#           copy_file_to_gcs(self.job_dir, checkpoints[-1])
#       else:
#         print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))