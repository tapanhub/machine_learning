from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
import os
import traceback

logdir = "/tmp/log"


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
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


for fl in os.listdir(logdir):
    logfile = os.path.join(logdir, fl)
    try:
        print(f"log file: {logfile}")
        read_eventfile(logfile)
    except Exception as e:
        print(f"read_eventfile failed with {str(e)}")
        print(traceback.format_exc())
