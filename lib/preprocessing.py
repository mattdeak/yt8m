import os
import time
import multiprocessing
import numpy as np


def _file_reader(num):
    if num < 10:
        num = f'0{num}'
    elif num > 90:
        num = 9000 + num - 90
    out = []
    with open(f'/home/data/glove/x{num}', 'rb') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0].decode()
            embedding = np.array([float(val) for val in splitLine[1:]])
            out.append((word, embedding))
    return out


def load_glove():
    start_time = time.time()
    p = multiprocessing.Pool(multiprocessing.cpu_count()-2)
    results = p.map(_file_reader, range(len(os.listdir('/home/data/glove'))))
    glove = dict([pair for result in results for pair in result])
    print(f"Glove vectors loaded in {time.time() - start_time} seconds")
    return glove
