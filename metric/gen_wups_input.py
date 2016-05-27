import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math


predict_file = json.load(open('cocoqa_lstm_results_vgg.json', 'r'))
gt_file = json.load(open('data/coco_qa_raw_test.json', 'r'))


# calculate the accuracy for each type:
acc = 0
acc0 = 0
acc1 = 0
acc2 = 0
acc3 = 0

count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
f1 = open('gt_ans_save.txt', 'w')
f2 = open('pd_ans_save.txt', 'w')

for i in range(len(predict_file)):
    pre_id = predict_file[i]['question_id']
    gt_id = gt_file[i]['ques_id']

    if not pre_id == gt_id:
        raise AssertionError()

    pre_ans = predict_file[i]['answer'].lower()
    gt_ans = gt_file[i]['ans'].lower()

    ques_type = int(gt_file[i]['types'])
    if pre_ans == gt_ans:
        acc += 1
    count += 1
    if ques_type == 0:
        if pre_ans == gt_ans:
            acc0 += 1
        count0 += 1
    elif ques_type == 1:
        if pre_ans == gt_ans:
            acc1 += 1
        count1 += 1
    elif ques_type == 2:
        if pre_ans == gt_ans:
            acc2 += 1
        count2 += 1
    elif ques_type == 3:
        if pre_ans == gt_ans:
            acc3 += 1
        count3 += 1


    # write the gt and answer
    f1.write(gt_ans + '\n')
    f2.write(pre_ans + '\n')

f1.close()
f2.close()
prob = float(acc) / float(count)
prob0 = float(acc0) / float(count0)
prob1 = float(acc1) / float(count1)
prob2 = float(acc2) / float(count2)
prob3 = float(acc3) / float(count3)
print('total Acc %f' %prob)
print('Acc 0 %f' % prob0)
print('Acc 1 %f' % prob1)
print('Acc 2 %f' % prob2)
print('Acc 3 %f' % prob3)

