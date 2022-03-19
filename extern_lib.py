import tensorflow as tf
import numpy as np
from tensorflow import keras, nn, losses, math
from tensorflow.keras import Sequential, layers, datasets, optimizers, Model
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import random


