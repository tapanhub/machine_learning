from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
import os
import traceback
import glob

logdir = "/tmp/log_"


def read_eventfile(sfile):
    # tf version 1.x
    # use tf.python.summary.summary_iterator(sfile)
    # for 2.x
    for event in tf.compat.v1.train.summary_iterator(sfile):
        for value in event.summary.value:
            print("---------tag--------\n")
            print(value.tag)
            print("---------val--------\n")
            print(value)
            if value.HasField('simple_value'):
                print("---------simple_value--------\n")
                print(value.simple_value)


# write dummy data to tensorboard
with SummaryWriter(logdir) as writer:
    for n_iter in range(100):
        rand_vals=[ np.random.random() for _ in range(4) ]
        print(rand_vals)
        writer.add_scalar('Loss/train', rand_vals[0], n_iter)
        writer.add_scalar('Loss/test', rand_vals[1], n_iter)
        writer.add_scalar('Accuracy/train', rand_vals[2], n_iter)
        writer.add_scalar('Accuracy/test', rand_vals[3], n_iter)
        tdict= {
                 'Loss/train': rand_vals[0],
                 'Loss/test': rand_vals[1],
                 'Accuracy/train': rand_vals[2],
                 'Accuracy/test': rand_vals[3],
               }
        writer.add_scalars('loss_accuracy/train_test', tdict, n_iter)


files = glob.glob(logdir + '/**/*',
                  recursive = True)
for logfile in files:
    if os.path.isfile(logfile):
        try:
            print(f"log file: {logfile}")
            read_eventfile(logfile)
        except Exception as e:
            print(f"read_eventfile failed with {str(e)}")
            print(traceback.format_exc())
