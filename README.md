# ISPL_Freshman_practice

<MNIST classification>

mnist_done에 학습에 사용한 train, val, test와 tfrecord 파일이 압축되어 있습니다.

result.txt에 학습 과정과 결과를 기록하였습니다.

usage: 

    python main.py --process=write --imagedir=./mnist/train --datadir=./mnist/train_tfrecord

    or

    python main.py --process=train --datadir=./mnist/train_tfrecord --val_datadir=./mnist/val_tfrecord --epoch=1 --lr=1e-3 --ckptdir=./ckpt --batch=100 --restore=False
