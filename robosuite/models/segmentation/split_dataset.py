import os
import random

def split_by_type(file, path, type=4):
    lines = open(file).readlines()
    fs = []
    for t in range(type):
        p = os.path.join(path, 'label_' + str(t) + '.txt')
        fs.append(open(p, 'w'))

    for line in lines:
        path, action, obj_type, reward = line.split()
        f = fs[int(obj_type)]
        s = path + ' ' + action + ' ' + obj_type + ' ' + reward
        f.write(s + '\n')

    for f in fs:
        f.close()


def split_train_test(file, ratio=0.8):
    lines = open(file).readlines()
    num = len(lines)
    random.shuffle(lines)

    train_num = int(num * ratio)

    train_lines = lines[:train_num]
    test_lines = lines[train_num:]

    assert len(train_lines) > len(test_lines)

    path = file.split('.')[0]
    train_path = path + '_train.txt'
    test_path = path + '_test.txt'

    train_file = open(train_path, 'w')
    test_file = open(test_path, 'w')

    for line in train_lines:
        train_file.write(line + '\n')
    for line in test_lines:
        test_file.write(line + '\n')

    train_file.close()
    test_file.close()


if __name__ == '__main__':
    split_by_type('/home/yeweirui/data/8obj_half_in_1m/label.txt', '/home/yeweirui/data/8obj_half_in_1m/')
    split_train_test('/home/yeweirui/data/8obj_half_in_1m/label_0.txt')
    split_train_test('/home/yeweirui/data/8obj_half_in_1m/label_1.txt')
    split_train_test('/home/yeweirui/data/8obj_half_in_1m/label_2.txt')
    split_train_test('/home/yeweirui/data/8obj_half_in_1m/label_3.txt')