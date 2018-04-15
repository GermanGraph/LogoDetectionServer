# The MIT License (MIT)
# Copyright (c) 2016 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import common
import model
import argparse
import os
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import util
import preprocess
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_fn', help='image filename')
    return parser.parse_args()


def logo_recognition(sess, img, obj_proposal, graph_params):
    # recognition results
    recog_results = {}
    recog_results['obj_proposal'] = obj_proposal

    # Resize image
    if img.shape != common.CNN_SHAPE:
        img = imresize(img, common.CNN_SHAPE, interp='bicubic')

    # Pre-processing
    img = preprocess.scaling(img)
    img = img.reshape((1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                       common.CNN_IN_CH)).astype(np.float32)

    # Logo recognition
    pred = sess.run(
        [graph_params['pred']], feed_dict={graph_params['target_image']: img})
    recog_results['pred_class'] = common.CLASS_NAME[np.argmax(pred)]
    recog_results['pred_prob'] = np.max(pred)

    return recog_results


def setup_graph():
    graph_params = {}
    graph_params['graph'] = tf.Graph()
    with graph_params['graph'].as_default():
        model_params = model.params()
        graph_params['target_image'] = tf.placeholder(
            tf.float32,
            shape=(1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
                   common.CNN_IN_CH))
        logits = model.cnn(
            graph_params['target_image'], model_params, keep_prob=1.0)
        graph_params['pred'] = tf.nn.softmax(logits)
        graph_params['saver'] = tf.train.Saver()
    return graph_params

def init_tf():
    # Setup computation graph
    graph_params = setup_graph()

    # Model initialize
    sess = tf.Session(graph=graph_params['graph'])
    tf.global_variables_initializer()
    if os.path.exists('models'):
        save_path = os.path.join('models', 'deep_logo_model')
        graph_params['saver'].restore(sess, save_path)
        print('Model restored')
    else:
        print('Initialized')
    return graph_params, sess

def main(file_name='', graph_params={}, sess={}):
    #args = parse_cmdline()
    img_fn = os.path.join("images", file_name)
    if not os.path.exists(img_fn):
        print('Not found: {}'.format(img_fn))
        sys.exit(-1)
    else:
        print('Target image: {}'.format(img_fn))

    # Load target image
    target_image = util.load_target_image(img_fn)
    #cv.normalize(target_image, target_image, 0, 255, cv.NORM_MINMAX)
    # limg = np.arcsinh(target_image)
    # limg /= limg.max()
    # low = np.percentile(limg, 0.25)
    # high = np.percentile(limg, 99.5)
    # opt_img = skie.exposure.rescale_intensity(limg, in_range=(low, high))
    # target_image = opt_img
    # target_image = target_image.astype(np.float64)

    # Get object proposals
    object_proposals = util.get_object_proposals(target_image)    

    # Logo recognition
    results = []
    for obj_proposal in object_proposals:
        x, y, w, h = obj_proposal
        crop_image = target_image[y:y + h, x:x + w]
        results.append(
            logo_recognition(sess, crop_image, obj_proposal, graph_params))

    del_idx = []
    for i, result in enumerate(results):
        if result['pred_class'] == common.CLASS_NAME[-1]:
            del_idx.append(i)
    results = np.delete(results, del_idx)

    # Non-max suppression
    nms_results = util.nms(results, pred_prob_th=0.9, iou_th=0.4)

    # Draw rectangles on the target image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(target_image)
    for result in nms_results:
        print(result)
        (x, y, w, h) = result['obj_proposal']
        ax.text(
            x,
            y,
            "{} {:.2f}".format(result['pred_class'], result['pred_prob']),
            fontsize=13,
            bbox=dict(facecolor='red', alpha=0.7))
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, bbox_inches='tight', pad_inches=0)
    img.seek(0)
    return img
    

if __name__ == '__main__':
    main()
