import argparse
import logging
import os
from google.cloud import storage 
import tempfile
import dill as dpickle
import pandas as pd 
from tensorflow import keras 
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

project = 'machinaseptember2016'
bucket = 'mmelnick_kftest'
model_id = 'lstm_v1'
create_time = '12345'
nDay = 0

# Define some hyperparameters
batch_size = 100
nn_l1 = 64
nn_l2 = 32
nn_l3 = 16

#Fetch all the data we created in the preprocessing step

class Model:
    def __init__(self, bucket, model_id, create_time, 
        project = 'machinaseptember2016', 
        nDay = 0):
        self.project = project
        self.bucket = bucket
        self.model_id = model_id
        self.create_time = create_time
        self.nDay = nDay

    def list_blobs_with_prefix(self, bucket_name, prefix, delimiter=None):
        """Lists all the blobs in the bucket that begin with the prefix.

        This can be used to list all blobs in a "folder", e.g. "public/".

        The delimiter argument can be used to restrict the results to only the
        "files" in the given "folder". Without the delimiter, the entire tree under
        the prefix is returned. For example, given these blobs:

            /a/1.txt
            /a/b/2.txt

        If you just specify prefix = '/a', you'll get back:

            /a/1.txt
            /a/b/2.txt

        However, if you specify prefix='/a' and delimiter='/', you'll get back:

            /a/1.txt

        """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        bList = []
        for blob in blobs:
            bList.append(blob.name)
            # print(blob.name)
        pList = []
        if delimiter:
            for prefix in blobs.prefixes:
                pList.append(prefix)
                # print(prefix)
        return bList, pList

    def list_nday_files(self):
        #Fetch number of predictions (folders)
        self.topDirString = self.model_id + '-' + self.create_time + '/'
        print('Checking number of folders in ' + self.topDirString)
        bList, pList = self.list_blobs_with_prefix(self.bucket, self.topDirString, delimiter='/')

        print(str(len(pList)) + ' timepoint folder(s) found, checking for files')
        if len(pList) == 0:
            raise(ValueError, 'No folders found! Check your model_id and current time and verify they are in GCS')
        elif len(pList) >= nDay:
            bList, pList = self.list_blobs_with_prefix(self.bucket, self.topDirString + str(nDay), delimiter='')
        #Stores list of files in self structure
        self.availableFiles = bList
    
    def download_blob(self, bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))

    def fetch_nday_files(self):
        self.temp_dir = tempfile.mkdtemp()
        for f in self.availableFiles:
            self.download_blob(self.bucket, 
                                f,
                                os.path.join(self.temp_dir, f.split('/')[-1]))            
    def load_fetched_files(self):
        from numpy import load
        files = os.listdir(self.temp_dir)
        loaded_vars = {}
        for file in files:
            if file.split('.')[-1] == 'dpkl':
                with open(os.path.join(self.temp_dir, file), 'rb') as f:
                    loaded_vars[file.split('.')[0]] = dpickle.load(f)  
            elif file.split('.')[-1] == 'npy':
                loaded_vars[file.split('.')[0]] = load(os.path.join(self.temp_dir, file))
            else:
                print('Warning, found files that were neither pickles nor serialized numpy arrays')  
        self.loaded_vars = loaded_vars
        logging.info('Successfully retrieved serialized data files')


    def build_model(self):
        """Build a keras model."""
        logging.info("starting to build model")
        # if self.job_name and self.job_name.lower() in ["ps"]:
        #   logging.info("ps doesn't build model")
        #   return
        tX = self.loaded_vars['trainX']
        model = keras.Sequential()
        model.add(keras.layers.LSTM(nn_l1, input_shape=(tX.shape[1], 
                            tX.shape[2]), 
                            return_sequences = True))
        model.add(keras.layers.LSTM(nn_l2, return_sequences = True))
        model.add(keras.layers.LSTM(nn_l3, ))
        model.add(keras.layers.Dense(1))
        self.keras_model = model 
        self.keras_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        self.keras_model.summary()

    def train_keras(self,
                    base_name= model_id, 
                    batch_size= batch_size, 
                    epochs=300):
        """Train using Keras.

        """
        logging.info("Using base name: %s", base_name)
        csv_logger = keras.callbacks.CSVLogger('{:}.log'.format(base_name))
        model_checkpoint = keras.callbacks.ModelCheckpoint(
        '{:}.epoch{{epoch:02d}}-val{{loss:.5f}}.hdf5'.format(
            base_name), save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor = 'loss',
                                                        min_delta=0,
                                                    patience=5,
                                                    verbose=1,
                                                    mode='auto')
        
        tX = self.loaded_vars['trainX']
        tY = self.loaded_vars['trainY']

        self.history = self.keras_model.fit(tX, 
                                        tY,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    callbacks=[csv_logger, 
                                                model_checkpoint,
                                                early_stopping])

        #############
        # Save model.
        #############
        self.keras_model.save(base_name + 'keras_model.h5')

    def evaluate_keras(self):
        from numpy import concatenate, tile
        """Generates predictions on holdout set and calculates mse ."""
        # 
        predict_lag = self.loaded_vars['jobData']['predict_lag']
        currLookBack = self.loaded_vars['jobData']['currLookBack']
        predict_inputs = self.loaded_vars['predict_inputs']
        tX = self.loaded_vars['trainX']
        tY = self.loaded_vars['trainY']
        dfNext = self.loaded_vars['dfNext']
        self.trueVals = dfNext.values
        scal = self.loaded_vars['scaler']
        nCols = self.loaded_vars['jobData']['nCols']
        #Reform allData variable from reshaped trainX and trainY
        allData = concatenate((tX.reshape((tX.shape[0], tX.shape[2])), tY), axis = 1)
        minPredictions = pd.DataFrame()
        for ii in range(0, 390-predict_lag):
            if ii < currLookBack:
                thisRow = concatenate(
                    (allData[-currLookBack+ii:, :-1], predict_inputs.values[:ii, :-1]))

            else:
                thisRow = predict_inputs.values[ii-currLookBack:ii, :-1]
            # Reshape
            thisRow = thisRow.reshape((thisRow.shape[0], 1, thisRow.shape[1]))

            # make predictions
            trainPredict = self.keras_model.predict(thisRow)
            # testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scal.inverse_transform(tile(trainPredict, [1, nCols]))
            minPredictions = minPredictions.append(
                pd.Series(trainPredict[-1][0]), ignore_index=True)
        # Tag with the proper timestamps
        minPredictions.index = dfNext.index[predict_lag:]
        #This should maybe print an error measurement, that or we need
        #to plan to deploy these jobs and concatenate later 
        self.minPredictions = minPredictions

    # create metric for hyperparameter tuning
    def my_rmse(self, ts1, ts2):
    #     from tensorflow.metrics import root_mean_squared_error
    #     return root_mean_squared_error(ts1, ts2)
        from numpy import sqrt, mean
        return sqrt(mean(ts1-ts2)**2)

    def write_out_metric(self, output_dir, rmse):
        from tensorflow.core.framework.summary_pb2 import Summary
        summary = Summary(value=[Summary.Value(tag='rmse', 
                                        simple_value=rmse)])
        eval_path = os.path.join(output_dir, 'rmse')
        summary_writer = tf.summary.FileWriter(eval_path)

        # Note: adding the summary to the writer is enough for hyperparameter tuning.
        # AI Platform looks for any summary added with the hyperparameter metric tag.
        summary_writer.add_summary(summary)
        summary_writer.flush()

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print('File {} uploaded to {}.'.format(
            source_file_name,
            destination_blob_name))
    def save_predictions(self):
        temp_dir = tempfile.mkdtemp()
        fName = os.path.join(temp_dir, 'predictions.csv')
        self.minPredictions.to_csv(fName)
        self.upload_blob(self.bucket, 
                            fName, 
                            self.model_id + '-' + self.create_time + '/' + str(self.nDay) + '/predictions.csv')
