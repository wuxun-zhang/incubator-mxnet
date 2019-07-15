#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from __future__ import print_function
import sys, os
import argparse
import subprocess
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from pascal_voc import VocSegmentation
from mscoco import CocoSegmentation

def load_voc():
    pass


def load_coc():
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2012', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval',
                        default='trainval', type=str)
    parser.add_argument('--target-dir', dest='target_dir', help='output list directory',
                        default=os.path.join(curr_path, 'dataset'),
                        type=str)
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--no-shuffle', dest='shuffle', help='shuffle list',
                        action='store_false')
    parser.add_argument('--num-thread', dest='num_thread', type=int, default=1,
                        help='number of thread to use while runing im2rec.py')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    target = os.path.join(args.target_dir, args.set + '.lst')
    if args.dataset == 'pascal':
        db = VocSegmentation(args.set, args.year, args.root_path, args.shuffle)
        print("saving list to disk...")
        db.save_imglist(target, root=args.root_path)
    elif args.dataset == 'coco':
        db = CocoSegmentation(args.set, args.root_path, args.shuffle)
        print("saving list to disk...")
        db.save_imglist(target, root=args.root_path)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)

    print("List file {} generated...".format(target))

    cmd_arguments = ["python",
                    os.path.join(curr_path, "../../tools/im2rec.py"),
                    os.path.abspath(target), os.path.abspath(args.root_path),
                    "--pack-label", "--num-thread", str(args.num_thread)]

    if not args.shuffle:
        cmd_arguments.append("--no-shuffle")

    subprocess.check_call(cmd_arguments)

    print("Record file {} generated...".format(target.split('.')[0] + '.rec'))


