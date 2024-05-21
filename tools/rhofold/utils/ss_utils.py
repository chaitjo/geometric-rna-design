# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np

def preprocess_ss_map(prob_map, seq, threshold=0.5, nc=True):
    canonical_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

    # candidate 1: threshold
    contact = (prob_map > threshold)
    prob_map = prob_map * (1 - np.eye(prob_map.shape[0]))

    x_array, y_array = np.nonzero(contact)
    prob_array = []
    for i in range(x_array.shape[0]):
        prob_array.append(prob_map[x_array[i], y_array[i]])
    prob_array = np.array(prob_array)

    sort_index = np.argsort(-prob_array)

    mask_map = np.zeros_like(contact)
    already_x = set()
    already_y = set()
    for index in sort_index:
        x = x_array[index]
        y = y_array[index]

        seq_pair = seq[x] + seq[y]
        if seq_pair not in canonical_pairs and nc == True:
            continue
            pass

        if x in already_x or y in already_y:
            continue
        else:
            mask_map[x, y] = 1
            already_x.add(x)
            already_y.add(y)

    contact = contact * mask_map
    return contact

def save_ss2ct(prob_map, seq, save_file, threshold=0.5):
    """
    :param contact: binary matrix numpy
    :param seq: string
    :return:
    generate ct file from ss npy
    """
    seq_len = len(seq)

    contact = preprocess_ss_map(prob_map, seq, threshold)

    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len+1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len+2))
    fifth_col = [pair_dict[i]+1 for i in range(seq_len)]
    last_col = list(range(1, seq_len+1))

    save_dir, _ = os.path.split(save_file)
    if os.path.exists(save_dir) != True:
        os.makedirs(save_dir)

    with open(save_file, "w") as f:
        f.write("{}\n".format(seq_len))
        for i in range(seq_len):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(first_col[i], second_col[i], third_col[i], fourth_col[i], fifth_col[i], last_col[i]))

    return contact