import numpy as np
import itertools
import copy
import random
import os

def get_hard_cases(path='data', trainFile='hard_case_train.npy', testFile='hard_case_test.npy'):

    trainFile = os.path.join(path, trainFile)
    testFile = os.path.join(path, testFile)

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(trainFile) and os.path.exists(testFile):
        hard_cases_train = np.load(trainFile)
        hard_cases_test = np.load(testFile)

        return hard_cases_train, hard_cases_test

    hard_cases = []
    z = 0.135
    train_ratio = 0.8

    def generate_case(x, y):
        poses = []

        poses.append(np.array([x, y, 0]))
        poses.append(np.array([x, -y, 0]))
        poses.append(np.array([-x, y, 0]))
        poses.append(np.array([-x, -y, 0]))

        poses.append(np.array([0, 0, z]))

        return poses

    xs = [0.02, 0.025, 0.03, 0.035, 0.04]
    ys = [0.02, 0.025, 0.03, 0.035]

    for x in xs:
        for y in ys:
            hard_cases.append(generate_case(x, y))


    hard_cases = np.array(hard_cases)
    np.random.shuffle(hard_cases)

    slice = int(len(hard_cases) * train_ratio)
    hard_cases_train = hard_cases[: slice]
    hard_cases_test = hard_cases[slice: ]

    np.save(trainFile, hard_cases_train)
    np.save(testFile, hard_cases_test)

    return hard_cases_train, hard_cases_test