# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import time
import sys

import mxnet as mx
from mxnet import gluon, profiler
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset
from gluoncv.utils.viz import get_color_pallete
from gluoncv.data.segbase import ms_batchify_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FCN model')
    # model and dataset 
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--image-shape', type=int, default=480,
                        help='image shape')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--quantized', action='store_true', 
                        help='whether to use quantized model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained model from GluonCV')
    parser.add_argument('--benchmark', action='store_true',
                        help='using dummy data for inference')

    args = parser.parse_args()
    return args
    

def evaluate(model, input_shape, ctx, num_workers, dataset, logger):
    bs = input_shape[0]
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    testset = get_segmentation_dataset(
            dataset, split='val', mode='val', transform=input_transform)
    size = len(testset)
    logger.info("The length of validation dataset is %d " % size)

    test_data = gluon.data.DataLoader(
            testset, bs, last_batch='rollover', shuffle=False, num_workers=num_workers)
    logger.info("Batch size for inference is %d" % bs)
    # warm up
    dry_run = 0
    data = [mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx, dtype='float32')]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run):
        outputs = model.forward(batch.data[0])
        for output in outputs:
            output.wait_to_read()

    # model = get_model('fcn_resnet101_voc_int8', pretrained=True)
    # model.load_parameters('fcn_resnet101_voc_int8-0000.params', ctx=ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)
    tbar = tqdm(test_data)

    metric.reset()
    tic = time.time()
    for i, (batch, dsts) in enumerate(tbar, 0):
        targets = mx.gluon.utils.split_and_load(dsts, ctx_list=[ctx], even_split=False)
        data = batch.as_in_context(ctx)
        outputs = model.forward(data)
        metric.update(targets, outputs)
        pixAcc, mIoU = metric.get()
        tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    
    assert i+1 == size//bs, "The length of validation dataset should be checked carefully"
    speed = size / (time.time() - tic)
    logger.info('Throughput is %f img/sec' % speed)

def benchmark(model, input_shape, ctx, num_batches, logger):
    bs = input_shape[0]
    size = num_batches * bs
    data = [mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx, dtype='float32')]
    batch = mx.io.DataBatch(data, [])

    # set profiler
    profiler.set_config(profile_all=True,
                        aggregate_stats=True,
                        filename='profile.json')

    dry_run = 5
    with tqdm(total=size) as pbar:
        for n in range(dry_run + num_batches):
            if n == dry_run:
                # profiler.set_state('run')
                tic = time.time()
            outputs = model.forward(batch.data[0])
            for output in outputs:
                output.wait_to_read()
            pbar.update(bs)
        # profiler.set_state('stop')
    # print(profiler.dump_profile())
    speed = size / (time.time() - tic)
    logger.info('Throughput is %f imgs/sec' % speed)


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    ctx = mx.cpu(0)
    bs = args.batch_size
    CHANNEL_COUNT = 3
    num_batches = args.num_batches
    image_shape = args.image_shape
    input_shape = (bs, CHANNEL_COUNT, image_shape, image_shape)

    if args.backbone not in ['resnet101', 'resnet50']:
        raise ValueError('Unsupported base network {} for fcn'.format(args.backbone)) 

    model_prefix = args.model + '_' + args.backbone
    if 'pascal' in args.dataset:
        model_prefix += '_voc'
    elif args.dataset == 'coco':
        model_prefix += '_coco'
    elif args.dataset == 'ade20k':
        model_prefix += '_ade'
    else:
        raise ValueError('Unsupported dataset {} used'.format(args.dataset))

    if args.quantized:
        model_prefix += '_int8'

    model = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(model_prefix), ['data'], 
            '{}-0000.params'.format(model_prefix))
    # if args.pretrained:
    #     model = get_model(model_prefix, pretrained=True)
    # else:
    #     model = get_model(model_prefix, pretrained=False)
    #     model.load_parameters(model_prefix)
    logger.info("Successfully loaded %s model." % model_prefix)
    model.collect_params().reset_ctx(ctx = ctx)
    model.hybridize()

    if not args.benchmark:
        evaluate(model, input_shape, ctx, args.num_workers, args.dataset, logger)
    else:
        benchmark(model, input_shape, ctx, num_batches, logger)
    logger.info('Evaluation on FCN model has been done!')
