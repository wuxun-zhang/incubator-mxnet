'''Parser for MXNet-based benchmark Opperf output'''
import json
import argparse

def add_args(parser):
    parser.add_argument("--file",
                        help="benchmark json file", default="mxnet_operator_benchmarks.json")
    parser.add_argument("--skipped-op", help="operator don't need to anlyze", default="")
    parser.add_argument("--skipped-matched-op", help="operator (wild match) don't need to anlyze", default="")
    parser.add_argument("--item", help="item (wild matched) need to be printed", default='all')


def print_each_item(args, d, item):
    print("----------- Start parsing %s ----------" % (item))
    for op_name, rst_dict_list in d.items():
        # print('-- %s ---' % (op_name))
        if args.skipped_op == op_name or \
            (args.skipped_matched_op is not '' and args.skipped_matched_op in op_name):
            continue
        no_output = True
        for rst_dict in rst_dict_list:
            for k, v in rst_dict.items():
                assert isinstance(k, str), "key's type here should be string!"
                if item in k:
                    if no_output:
                        no_output = False
                    print(rst_dict[k])
            if no_output:
                print('---')
    print('----------- End parsing %s -----------' % (item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mxnet benchmark opperf parser with CPU backend',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()

    f = open(args.file, 'r')
    j = json.load(f)

    # print(j.items())
    items = ['forward', 'backward', 'mem_alloc', 'p50', 'p90', 'p99']
    if args.item == 'all':
        for item in items:
            print_each_item(args, j, item)
    else:
        print_each_item(args, j, args.item)


