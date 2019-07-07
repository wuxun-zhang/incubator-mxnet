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

import argparse
import os
import sys
import logging
import mxnet as mx
import gluoncv
from mxnet import gluon, ndarray as nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv import data as gdata
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
# from mxnet.contrib.quantization import *
from quantize_model import *

def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


def get_calib_dataset(dataset, transform):
    calib_dataset = get_segmentation_dataset(
            dataset, split='val', mode='val', transform=transform)
    return calib_dataset


def get_dataloader(calib_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_loader = gluon.data.DataLoader(calib_dataset, batch_size=batch_size, shuffle=False,
            last_batch='rollover', num_workers=num_workers)
    return val_loader



def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='backbone name (default: resnet101)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 16)')
    parser.add_argument('--num-calib-batches', type=int, default=5,
                        help='number of batches for calibration')
    parser.add_argument('--calib-mode', type=str, default='entropy',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--enable-calib-quantize', type=bool, default=True,
                        help='If enabled, the quantize op will '
                             'be calibrated offline if calibration mode is '
                             'enabled')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    ctx = mx.cpu(0)

    if 'pascal' in args.dataset:
        model_prefix = args.model + '_' + args.backbone + '_voc'
    elif args.dataset == 'coco':
        model_prefix = args.model + '_' + args.backbone + '_coco'
    else:
        raise ValueError('%s dataset is not supported yet' % args.dataset)


    # get model from gluonCV.ModelZoo
    net = get_model(model_prefix, pretrained=True)
    # net.hybridize()
    # net(mx.nd.ones(shape=(1, 3, 224, 224)))
    # net.export(model_prefix, 0)
    gluoncv.utils.export_block(model_prefix, net, preprocess=False, layout='CHW')
    # x = mx.sym.Variable('data')
    # y = net(x)
    # print(type(net))
    logger.info('Successfully saved symbol and params into files')

    # sys.exit()

    # get config
    batch_size = args.batch_size
    num_calib_batches = args.num_calib_batches
    logger.info('batch size = %d for calibration' % batch_size)
    logger.info('sampling %d batches for calibration' % num_calib_batches)

    logger.info('load model %s' % model_prefix)
    calib_mode = args.calib_mode
    calib_mode = 'naive'
    logger.info('calibration mode set to %s' % calib_mode)

    # load model back
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    logger.info('Successfully loaded symbol and params from file')

    # 
    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')

    calib_layer = lambda name: name.endswith('_output') or name == "data"

    excluded_sym_names = []

    if calib_mode == None:
        raise ValueError('%s is not supported yet' % args.calib_mode)
    else:
        logger.info('Creating GluonDataLoader for segmentation dataset')
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        calib_dataset = get_calib_dataset(args.dataset, input_transform)
        calib_loader = get_dataloader(calib_dataset, batch_size=batch_size, 
                                      num_workers=args.data_nthreads)

        # data_shape = (batch_size, 3, 512, 512)
        data_shape = (batch_size, 3, args.crop_size, args.crop_size)
        qsym, qarg_params, aux_params = my_quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                          data_shape=data_shape,
                                                          ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                          calib_mode=calib_mode, calib_data=calib_loader,
                                                          num_calib_batch=num_calib_batches,
                                                          calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                          logger=logger)

        if calib_mode == 'naive':
            suffix = '_int8'
        elif calib_mode == 'entropy':
            suffix = '_int8'
        sym_name = '%s-symbol.json' % (model_prefix + suffix)
    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, qsym, logger=logger)
    params_name = '%s-%04d.params' % (model_prefix + suffix, 0)
    save_params(params_name, qarg_params, aux_params, logger=logger)
