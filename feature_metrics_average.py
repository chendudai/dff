import os.path
import os
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, f1_score
from collections import namedtuple, Counter
Metrics = namedtuple("Metrics", "ap ba jaccard dice label")



class MetricCalculator:
    
    GT_THRESHOLD = 0.5
    PRED_THRESHOLD = 0.5

    def __init__(self, path_pred, path_gt, top_k, num_epochs):
        self.path_pred = path_pred
        self.path_gt = path_gt
        self.top_k = str(top_k)
        self.num_epochs = str(num_epochs)
    def process_single_image(self, pred_fn, gt_fn, label=None, verbose=False):

        pred_img = Image.open(os.path.join(self.path_pred, pred_fn)).convert('L')
        gt_img = Image.open(os.path.join(self.path_gt, label, gt_fn)).convert('L')

        pred_array = 1 - np.asarray(pred_img) / 255
        gt_array = 1 - np.asarray(gt_img) / 255
        # 1 - x : assumes background is x=1

        gt_uniq = np.unique(gt_array)
        gt_nuniq = len(gt_uniq)
        if gt_nuniq != 2:
            print(f'Warning: Ground truth mask contains {gt_nuniq} unique values:', *gt_uniq)
            if verbose:
                print('Using threshold', self.GT_THRESHOLD)

        gt_mask = gt_array > self.GT_THRESHOLD

        gt_mask_flat = gt_mask.ravel()
        pred_array_flat = pred_array.ravel()

        pred_array_flat_thresh = pred_array_flat > self.PRED_THRESHOLD

        # calculate metrics
        AP = average_precision_score(gt_mask_flat, pred_array_flat)

        tpr = (gt_mask_flat * pred_array_flat).sum() / gt_mask_flat.sum()
        tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / (1 - gt_mask_flat).sum()

        balanced_acc = (tpr + tnr) / 2

        jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

        dice = f1_score(gt_mask_flat, pred_array_flat_thresh)
        
        metrics = Metrics(AP, balanced_acc, jscore, dice, label)

        # if verbose:
        with open(os.path.join(self.path_pred, pred_fn.split('.')[0] + '_metric.txt'),'w')as file:
            file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
            file.write(f'\tAP (average precision):\t {metrics.ap}\n')
            file.write(f'\tBalanced accuracy:\ {metrics.ba}\n')
            file.write(f'\tJaccard score (IoU):\t {metrics.jaccard}, Threshold: {self.PRED_THRESHOLD}\n')
            file.write(f'\tDice score (F1):\t {metrics.dice}, Threshold: {self.PRED_THRESHOLD}')
            file.close()

        # print(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})')
        # print('\tAP (average precision):\t', metrics.ap)
        # print('\tBalanced accuracy:\t', metrics.ba)
        # print('\tJaccard score (IoU):\t', metrics.jaccard, f'(Threshold: {self.PRED_THRESHOLD})')
        # print('\tDice score (F1):\t', metrics.dice, f'(Threshold: {self.PRED_THRESHOLD})')
        #
        return metrics
    
    def process_all_images(self, pred_fns, gt_fns, labels):
        assert len(pred_fns) == len(gt_fns) == len(labels), 'Mismatched number of filenames and/or labels'
        all_metrics = [
            self.process_single_image(pred_fn, gt_fn, label)
            for pred_fn, gt_fn, label in zip(pred_fns, gt_fns, labels)
        ]
        unique_labels = np.unique(labels)
        counts = Counter(labels)
        
        # print("Unique labels:", *unique_labels)
        # print("Counts:")
        # for label in unique_labels:
        #     print(f"\t{label}: {counts[label]}")
        # print()
        #

        with open(self.path_pred + '/all_category_metric.txt','w')as f:
            f.write("all_category_metric :\n")
            f.write('\tAP (average precision):\t' + str(np.mean([m.ap for m in all_metrics])) + '\n')
            f.write('\tBalanced accuracy:\t' + str(np.mean([m.ba for m in all_metrics])) + '\n')
            f.write('\tJaccard score (IoU):\t' + str(
                np.mean([m.jaccard for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\tDice score (F1):\t' + str(
                np.mean([m.dice for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\n')

        # print("Micro-averaged metrics (per-sample average):")
        # print('\tAP (average precision):\t', np.mean([m.ap for m in all_metrics]))
        # print('\tBalanced accuracy:\t', np.mean([m.ba for m in all_metrics]))
        # print('\tJaccard score (IoU):\t', np.mean([m.jaccard for m in all_metrics]), f'(Threshold: {self.PRED_THRESHOLD})')
        # print('\tDice score (F1):\t', np.mean([m.dice for m in all_metrics]), f'(Threshold: {self.PRED_THRESHOLD})')
        # print()
        #
        # print("Macro-averaged metrics (per-class average):")
        def macro_average(metric_name):
            values = [
                np.mean([getattr(m, metric_name) for m in all_metrics if m.label == label])
                for label in unique_labels
            ]
            return np.mean(values)
        #
        #
        # print('\tAP (average precision):\t', macro_average('ap'))
        # print('\tBalanced accuracy:\t', macro_average('ba'))
        # print('\tJaccard score (IoU):\t', macro_average('jaccard'), f'(Threshold: {self.PRED_THRESHOLD})')
        # print('\tDice score (F1):\t', macro_average('dice'), f'(Threshold: {self.PRED_THRESHOLD})')
        # print()

        return {
            k: macro_average(k) for k in ['ap', 'ba', 'jaccard', 'dice']
        }

def main_metrics(path_pred, path_gt, category, top_k, num_epochs):
    calculator = MetricCalculator(path_pred, path_gt, top_k, num_epochs)

    if category == 'window':
        pred_fns = ['017_semantic.png', '023_semantic.png', '029_semantic.png', '089_semantic.png', '117_semantic.png',
                    '131_semantic.png', '633_semantic.png']
        gt_fns = ['0017_mask.jpg', '0023_mask.JPG', '0029_mask.jpg', '0089_mask.JPG', '0117_mask.jpg', '0131_mask.jpg', '0633_mask.jpg']
        labels = ['window'] * 7
    elif category == 'portal':
        pred_fns = ['120_semantic.png', '146_semantic.png', '350_semantic.png', '561_semantic.png', '751_semantic.png', '761_semantic.png', '796_semantic.png']
        gt_fns = ['0120_mask.jpg', '0146_mask.jpg', '0350_mask.jpg', '0561_mask.jpg', '0751_mask.jpg', '0761_mask.jpg', '0796_mask.JPG']
        labels = ['portal'] * 7
    elif category == 'spires':
        pred_fns = ['156_semantic.png', '259_semantic.png', '415_semantic.png', '455_semantic.png', '473_semantic.png']
        gt_fns = ['0156_mask.jpg', '0259_mask.jpg', '0415_mask.jpg', '0455_mask.jpg', '0473_mask.JPG']
        labels = ['spires'] * 5
    elif category == 'facade':
        pred_fns = ['025_semantic.png', '634_semantic.png', '637_semantic.png', '645_semantic.png', '764_semantic.png']
        gt_fns = ['0025_mask.jpg', '0634_mask.JPG', '0637_mask.jpg', '0645_mask.JPG', '0764_mask.jpg']
        labels = ['facade'] * 5
    else:
        raise ValueError('no category')


    return calculator.process_all_images(pred_fns, gt_fns, labels)


if __name__ == '__main__':
    main_metrics()