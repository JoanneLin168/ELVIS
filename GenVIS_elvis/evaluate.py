import argparse

from vita.data.datasets.ytvis_api.ytvos import YTVOS
from vita.data.datasets.ytvis_api.ytvoseval import YTVOSeval


parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True, help='path to ground truth json file')
parser.add_argument('--dt', type=str, required=True, help='path to detection json file')
args = parser.parse_args()

gt_file = args.gt
dt_file = args.dt

cocoGt = YTVOS(gt_file)
cocoDt = cocoGt.loadRes(dt_file)

evaluator = YTVOSeval(cocoGt, cocoDt, 'segm')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()