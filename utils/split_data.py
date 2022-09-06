import os
import random
import sys


def write(annots, save_path):
    with open(save_path, 'w') as f:
        for annot in annots:
            f.write(annot)
            f.write('\n')

def main(data_dir, test_ratio=0.8, shuffle=False):
    with open(os.path.join(data_dir, 'annotations.txt')) as f:
        annots = [l.strip() for l in f.readlines()]

    # List
    idxs = list(range(len(annots)))
    if shuffle:
        random.shuffle(idxs)

    # Idxs
    at = int(test_ratio*len(idxs))
    train_idxs = idxs[:at]
    test_idxs  = idxs[at:]

    # Split
    train = [annots[i] for i in train_idxs]
    test  = [annots[i] for i in test_idxs]

    # Write
    write(train, os.path.join(data_dir, 'train.txt'))
    write(test, os.path.join(data_dir, 'test.txt'))




if __name__=='__main__':
    main(sys.argv[1], float(sys.argv[2]))
