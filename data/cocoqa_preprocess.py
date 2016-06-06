"""
Download the cocoqa data and preprocessing.

Version: 1.0
Contributor: Jiasen Lu
"""


# Download the VQA Questions from http://www.visualqa.org/download.html
import json
import os
import argparse
import pdb

def download_cocoqa():
    print('Downloading COCO-QA data from http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip')
    os.system('wget http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip -P zip/')

    # Unzip the annotations
    os.system('unzip zip/cocoqa-2015-05-17.zip -d coco_qa/')


def main(params):
    if params['download'] == 1:
        download_cocoqa()

    train = []
    test = []
    imdir='%s/COCO_%s_%012d.jpg'

    print 'Loading annotations and questions...'

    f = open('coco_qa/train/questions.txt', 'r')
    train_ques = []
    train_ques_id = []
    ques_id = 0
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]

        line += ' ?'
        train_ques.append(line)
        train_ques_id.append(ques_id)
        ques_id += 1
    f.close()

    f = open('coco_qa/train/img_ids.txt', 'r')
    img_ids = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        img_ids.append(line)
    f.close()


    f = open('coco_qa/train/answers.txt', 'r')
    answers = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        answers.append(line)
    f.close()

    f = open('coco_qa/train/types.txt', 'r')
    types = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        types.append(line)
    f.close()

    subtype = 'train2014'
    for i in range(len(train_ques_id)):
        question_id = train_ques_id[i]
        image_path = imdir%(subtype, subtype, int(img_ids[i]))
        question = train_ques[i]
        ans = answers[i]
        tp = types[i]

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'types': tp,'ans': ans})


    f = open('coco_qa/test/questions.txt', 'r')
    val_ques = []
    val_ques_id = []
    ques_id = 0
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]

        line += ' ?'
        val_ques.append(line)
        val_ques_id.append(ques_id)
        ques_id += 1
    f.close()

    f = open('coco_qa/test/img_ids.txt', 'r')
    img_ids = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        img_ids.append(line)
    f.close()


    f = open('coco_qa/test/answers.txt', 'r')
    answers = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        answers.append(line)
    f.close()

    f = open('coco_qa/test/types.txt', 'r')
    types = []
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]        
        types.append(line)
    f.close()

    subtype = 'val2014'
    for i in range(len(val_ques_id)):
        question_id = val_ques_id[i]
        image_path = imdir%(subtype, subtype, int(img_ids[i]))
        question = val_ques[i]
        ans = answers[i]
        tp = types[i]
        test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'types': tp,'ans': ans})

    json.dump(train, open('cocoqa_raw_train.json', 'w'))
    json.dump(test, open('cocoqa_raw_test.json', 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', default=1, type=int, help='Download and extract data from cocoqa')

    # input json  
    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)









