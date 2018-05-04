# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os
import glob
import numpy as np
import scipy.misc

from stylize_c import stylize_c
from stylize import stylize

import math
from argparse import ArgumentParser

from PIL import Image

import imageio

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# default arguments
CONTENT_WEIGHT = 10
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 800
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_folder',
            dest='content_folder', help='content image_folder',
            metavar='CONTENT_FOLDER', required=True)
    parser.add_argument('--style',
            dest='style', help='one style image',
            metavar='STYLE', required=True)
    parser.add_argument('--output_folder',
            dest='output_folder', help='output path folder',
            metavar='OUTPUT_FOLDER', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    source_folder = options.content_folder + '/*'
    source_files = sorted(glob.glob(source_folder), key=numericalSort)
    #source_files = sorted(glob.glob('frames/*'), key=numericalSort)
    style_image = [imread(options.style)]
    style_blend_weights = [1]
    initial_noiseblend = 1.0
    
    output_files = []
    for i in range(len(source_files)):
        output_files.append(options.output_folder + '/' + str(i) + '.jpg')


    for i in range(len(source_files)):
        if(i == 0):
            content_image = imread(source_files[i])
            target_shape = content_image.shape
            initial = content_image           
            for iteration, image in stylize(
                network=options.network,
                initial=initial,
                initial_noiseblend=initial_noiseblend,
                content=content_image,
                styles=style_image,
                preserve_colors=options.preserve_colors,
                iterations=options.iterations,
                content_weight=options.content_weight,
                content_weight_blend=options.content_weight_blend,
                style_weight=options.style_weight,
                style_layer_weight_exp=options.style_layer_weight_exp,
                style_blend_weights=style_blend_weights,
                tv_weight=options.tv_weight,
                learning_rate=options.learning_rate,
                beta1=options.beta1,
                beta2=options.beta2,
                epsilon=options.epsilon,
                pooling=options.pooling,
                print_iterations=options.print_iterations,
                checkpoint_iterations=options.checkpoint_iterations
            ):
                output_file = None
                combined_rgb = image
                if iteration is not None:
                    if options.checkpoint_output:
                        output_file = options.checkpoint_output % iteration
                else:
                    output_file = output_files[i]
                if output_file:
                    imsave(output_file, combined_rgb)
        else:
            content_image = imread(source_files[i])
            target_shape = content_image.shape
            prev_content_image = imread(source_files[i-1])
            prev_style_image = imread(output_files[i-1])
            initial = content_image
            for iteration, image in stylize_c(
                network=options.network,
                initial=initial,
                initial_noiseblend=initial_noiseblend,
                content=content_image,
                styles=style_image,
                preserve_colors=options.preserve_colors,
                iterations=options.iterations,
                content_weight=options.content_weight,
                content_weight_blend=options.content_weight_blend,
                style_weight=options.style_weight,
                style_layer_weight_exp=options.style_layer_weight_exp,
                style_blend_weights=style_blend_weights,
                tv_weight=options.tv_weight,
                learning_rate=options.learning_rate,
                beta1=options.beta1,
                beta2=options.beta2,
                epsilon=options.epsilon,
                pooling=options.pooling,
                print_iterations=options.print_iterations,
                checkpoint_iterations=options.checkpoint_iterations,
                prev_style_image=prev_style_image,
                prev_content_image=prev_content_image
            ):
                output_file = None
                combined_rgb = image
                if iteration is not None:
                    if options.checkpoint_output:
                        output_file = options.checkpoint_output % iteration
                else:
                    output_file = output_files[i]
                if output_file:
                    imsave(output_file, combined_rgb)


    images = []    
    for filename in output_files:
        images.append(imageio.imread(filename))
    imageio.mimsave('output.gif', images)               



def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
