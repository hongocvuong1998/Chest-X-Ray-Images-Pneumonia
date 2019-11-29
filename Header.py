import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import random as rng
from PIL import Image
import chainer
from chainer.backends import cuda
# from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
import chainer.datasets as datasets
from functools import partial
import time
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu, to_gpu
from chainer import initializers
from shutil import copyfile
import csv

AddressDatasetCSV=['./chest-xray-pneumonia/Training.csv',
                   './chest-xray-pneumonia/Evaluation.csv'
                  ]


