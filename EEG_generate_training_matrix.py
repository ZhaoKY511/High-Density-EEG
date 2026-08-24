#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys
import numpy as np
from EEG_feature_extraction import generate_feature_vectors_from_samples


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
    FINAL_MATRIX = None

    for x in os.listdir(directory_path):

        if not x.lower().endswith('.csv'):
            continue

        if 'test' in x.lower():
            continue
        try:
            name, state, _ = x[:-4].split('-')
        except:
            print('Wrong file name', x)
            sys.exit(-1)

        if state.lower() == 'positive':
            state = 2.

        elif state.lower() == 'neutral':
            state = 1.

        elif state.lower() == 'negative':
            state = 0.

        else:
            print('Wrong file name', x)
            sys.exit(-1)

        print('Using file', x)
        full_file_path = directory_path + '/' + x
        vectors, header = generate_feature_vectors_from_samples(file_path=full_file_path,
                                                                nsamples= 250,
                                                                period=1.0,
                                                                state=state,
                                                                remove_redundant=True,
                                                                cols_to_ignore=cols_to_ignore
                                                                )

        print('resulting vector shape for the file', vectors.shape)

        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

    print('FINAL_MATRIX', FINAL_MATRIX.shape)

    # 打乱
    np.random.shuffle(FINAL_MATRIX)

    # 保存文件
    np.savetxt(output_file, FINAL_MATRIX, delimiter=',',
               header=','.join(header),
               comments='')

    return None


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('arg1: input dir\narg2: output file')
        sys.exit(-1)
    directory_path = sys.argv[1]
    output_file = sys.argv[2]
    gen_training_matrix(directory_path, output_file, cols_to_ignore=-1)
