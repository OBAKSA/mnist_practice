# ISPL_Freshmen_practice

<MNIST classification>

1.

download all files including mnist.zip

DO NOT use tensorflow.examples.tutorials.mnist library

2.

fill TODOs

you should modify main.py code


usage: 

    python main.py --process=write --imagedir=./mnist/train --datadir=./mnist/train_tfrecord

    or

    python main.py --process=train --datadir=./mnist/train_tfrecord --val_datadir=./mnist/val_tfrecord --epoch=1 --lr=1e-3 --ckptdir=./ckpt --batch=100 --restore=False
