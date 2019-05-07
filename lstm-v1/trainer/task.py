import argparse
import json
import os

from .model import Model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'GCS path to data.',
        default = 'mmelnick_kftest'
    )
    # parser.add_argument(
    #     '--output_dir',
    #     help = 'GCS location to write checkpoints and export models',
    #     required = True
    # )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 100
    )
    parser.add_argument(
        '--save_predictions',
        help = 'Whether to save a CSV of values. Off by default for hyperparam tuning',
        type = int,
        default = 0
    )
    parser.add_argument(
        '--model_id',
        help = 'Same as used earlier to ensure data is loaded from GCS',
        type = str,
        default = 'lstm_v1'
    )
    parser.add_argument(
        '--create_time',
        help = 'Same as used earlier to ensure data is loaded from GCS',
        type = str,
        default = '12345'
    )
    parser.add_argument(
        '--nDay',
        help = 'Which day value to pull from GCS',
        type = int,
        default = 0
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--nn_l1',
        help = 'Number of neurons in 1st LSTM layer',
        type = int,
        default= 64
    )
    parser.add_argument(
        '--nn_l2',
        help = 'Number of neurons in 2nd LSTM layer',
        type = int,
        default= 32
    )
    parser.add_argument(
        '--nn_l3',
        help = 'Number of neurons in 3rd LSTM layer',
        type = int,
        default= 16
    )
  
    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    # output_dir = arguments.pop('output_dir')
    Model.bucket     = arguments.pop('bucket')
    Model.model_id     = arguments.pop('model_id')
    Model.create_time     = arguments.pop('create_time')
    Model.batch_size = arguments.pop('batch_size')
    Model.nn_l1 = arguments.pop('nn_l1')
    Model.nn_l2 = arguments.pop('nn_l2')
    Model.nn_l3 = arguments.pop('nn_l3')

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        'gs://',
        Model.bucket,
        Model.model_id + '-' + Model.create_time, 
        'models',
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    m = Model(Model.bucket, Model.model_id, Model.create_time)
    m.list_nday_files()
    m.fetch_nday_files()
    m.load_fetched_files()
    m.build_model()
    m.train_keras(base_name = m.model_id, 
                    batch_size = m.batch_size)
    m.evaluate_keras()
    metr = m.my_rmse(m.trueVals[-m.minPredictions.shape[0]:, -1], m.minPredictions.values)
    m.write_out_metric(output_dir, metr)  
    if args.save_predictions:  
        m.save_predictions()