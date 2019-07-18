
from argparse import ArgumentParser

import scipy.io as scio
import numpy as np


def _flatten(x):
    try:
        return x[0][0]
    except IndexError:
        return None


MAT_DIR = '../ZuCo/task{}/Matlab files/results{}.mat'
SUBJECTS = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH',
            'ZKW', 'ZMG', 'ZPH', 'ZDN']
WORD_FEATURES = ['nFixations', 'meanPupilSize', 'FFD',
                 'FFD_pupilsize', 'TRT', 'TRT_pupilsize', 'GD', 'GD_pupilsize',
                 'GPT', 'GPT_pupilsize']
OUTPUT_DIR = '../data/ZuCo/et_features{}'

parser = ArgumentParser()
parser.add_argument('--task-num', default='1')
args = parser.parse_args()

if args.task_num == '1':
    task_num, task_code, num_sentences = ('1', '_SR', 400)
elif args.task_num == '2':
    task_num, task_code, num_sentences = ('2', '_NR', 300)
else:
    task_num, task_code, num_sentences = ('3', '_TSR', 407)

print('\n--- Extracting ZuCo ET features for', task_code, '---')
sentences = [{} for _ in range(num_sentences)]
for subj_num, subj in enumerate(SUBJECTS):
    print('Extracting from subject', subj)
    mat_file = scio.loadmat(MAT_DIR.format(task_num, subj + task_code))
    for s_num, sentence_data in enumerate(mat_file['sentenceData'][0]):
        try:
            words = [w[0] for w in sentence_data['word']['content'][0]]
        except IndexError:
            print('Error on content for subject', subj, 'for sentence', s_num)
            continue

        if 'words' in sentences[s_num]:
            # check if they're really the same sentence
            if words != sentences[s_num]['words']:
                print('Sentence', s_num, 'content mismatch!')
                print(words)
                print(sentences[s_num]['words'])
                continue
        else:
            # for the first subject loaded.
            sentences[s_num]['words'] = words
            sentences[s_num].update(
                {f: np.zeros((len(SUBJECTS), len(words)))
                 for f in WORD_FEATURES})

        for f in WORD_FEATURES:
            values = list(map(_flatten, sentence_data['word'][f][0]))
            if len(words) != len(values):
                print('imbalance!')

            sentences[s_num][f][subj_num] = values

np.save(OUTPUT_DIR.format(task_code), sentences, allow_pickle=True)
print('Done, ET features saved to:', OUTPUT_DIR.format(task_code))
