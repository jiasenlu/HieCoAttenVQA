"""
Download the VGG and deep residual model to extract image features.

Version: 1.0
Contributor: Jiasen Lu
"""

import os
import argparse
import json
def download_VGG():
    print('Downloading VGG model from http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel')
    os.system('wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel')
    os.system('wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt')

def download_deep_residual():
    print('Downloading deep residual model from https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7')
    os.system('wget https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7')
    os.system('wget https://raw.githubusercontent.com/facebook/fb.resnet.torch/master/datasets/transforms.lua')

def main(params):
    if params['download']  == 'VGG':
        download_VGG()
    else:
        download_deep_residual()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', default='VGG', help='VGG or Residual')
    # input json  
    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
