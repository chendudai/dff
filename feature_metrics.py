from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, f1_score

import os
import re

import imageio
from collections import namedtuple, Counter
Metrics = namedtuple("Metrics", "ap ba jaccard dice label")

class MetricCalculator:
    
    GT_THRESHOLD = 0.5
    PRED_THRESHOLD = 0.5
    
    def __init__(self,path_pred):
        self.path_pred=path_pred
        pass

    def process_image(self, pred_fn, gt_fn, label=None, verbose=False):
        pred_img = Image.open(pred_fn).convert('L')
        gt_img = Image.open(gt_fn).convert('L')
        width, height = gt_img.size
        #print("gt_size_"+str(width)+"_"+str(height))
        #print("pred_img_"+str(pred_img.size[0])+"_"+str(pred_img.size[1]))
        gt_img = gt_img.resize((int(width/4),int(height/4)))
        #print("gt_size_resized_"+str(gt_img.size[0])+"_"+str(gt_img.size[1]))
        gt_img.save("gt.jpg")
        pred_img.show()
        gt_img.show()

        #pred_array = 1 - np.asarray(pred_img) / 255
        pred_array = np.asarray(pred_img) / 255
        gt_array = 1 - np.asarray(gt_img) / 255
        # 1 - x : assumes background is x=1
        #pred_array.show()
        #gt_array.show()
        
        gt_uniq = np.unique(gt_array)
        gt_nuniq = len(gt_uniq)
        if gt_nuniq != 2:
            print(f'Warning: Ground truth mask contains {gt_nuniq} values unique:', *gt_uniq)
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

        print('AP (average precision):\t', AP)
        print('Balanced accuracy:\t', balanced_acc)
        print('Jaccard score (IoU):\t', jscore, f'(Threshold: {self.PRED_THRESHOLD})')
        print('Dice score (F1):\t', dice, f'(Threshold: {self.PRED_THRESHOLD})')
        
        metrics = Metrics(AP, balanced_acc, jscore, dice, label)

        # if verbose:
        with open(os.path.join(self.path_pred, pred_fn.split('.')[0] + '_metric.txt'),'w')as file:
            file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
            file.write(f'\tAP (average precision):\t {metrics.ap}\n')
            file.write(f'\tBalanced accuracy:\ {metrics.ba}\n')
            file.write(f'\tJaccard score (IoU):\t {metrics.jaccard}, Threshold: {self.PRED_THRESHOLD}\n')
            file.write(f'\tDice score (F1):\t {metrics.dice}, Threshold: {self.PRED_THRESHOLD}')
            file.close()
        return metrics
        
    
    def process_all_images(self, pred_fns, gt_fns,labels):
        assert len(pred_fns) == len(gt_fns) == len(labels), 'Mismatched number of filenames and/or labels'
        all_metrics = [
            self.process_image(pred_fn, gt_fn, label)
            for pred_fn, gt_fn, label in zip(pred_fns, gt_fns,labels)
        ]
        
        with open(self.path_pred + '/all_category_metric.txt','w')as f:
            f.write("all_category_metric :\n")
            f.write('\tAP (average precision):\t' + str(np.mean([m.ap for m in all_metrics])) + '\n')
            f.write('\tBalanced accuracy:\t' + str(np.mean([m.ba for m in all_metrics])) + '\n')
            f.write('\tJaccard score (IoU):\t' + str(
                np.mean([m.jaccard for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\tDice score (F1):\t' + str(
                np.mean([m.dice for m in all_metrics])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\n')

        def macro_average(metric_name):
            values = [
                np.mean([getattr(m, metric_name) for m in all_metrics])
            ]
            return np.mean(values)
        return {
            k: macro_average(k) for k in ['ap', 'ba', 'jaccard', 'dice']
        }


if __name__ == "__main__":

    images=[]
    images_gt = []
    scores = []
    scores_gt = []
    mask_gt = []
    
    #gt_path = "/storage/chendudai/data/manually_gt_masks_badshahi/portals"
    #eval_path = "/storage/hanibezalel/Ha-NeRF/eval_hurba/results/phototourism/figure_out_dome_clean"
    #gt_path = "/storage/chendudai/data/manually_gt_masks_hurba/windows"
    #eval_path = "/storage/hanibezalel/Ha-NeRF/eval_st_paul_200/results/phototourism/figure_out_window_clean"
    #gt_path = "/storage/chendudai/data/manually_gt_masks_0_1/windows"
    #gt_path = "/storage/chendudai/data/manually_gt_masks_st_paul/towers"
    #gt_path = "/storage/chendudai/data/manually_gt_masks_notre_dame/portals"
    gt_path = "/storage/chendudai/data/manually_gt_masks_0209/windows"
    #eval_path = "/storage/hanibezalel/Ha-NeRF/eval_milano_200/results/phototourism/figure_out_portal_clean"
    #eval_path = "/storage/hanibezalel/Ha-NeRF/eval_blue_m_200/results/phototourism/figure_out_windows"
    #gt_path = "/storage/hanibezalel/Ha-NeRF/eval/results/phototourism/dff_sample_0"
    #eval_path = "/storage/hanibezalel/Ha-NeRF/eval/results/phototourism/dff_sample_0"
    eval_path = "/storage/hanibezalel/wikiscenes/logs/masks/blue_window/crf"
    
    #milan
    #ts_list = [17, 23, 29, 89, 117, 131, 633] # window,s
    #ts_list = [120, 146, 350, 561, 751, 761, 796] # portal,s
    #ts_list = [156, 259, 415, 455, 473] # spires
    #ts_list = [25, 634, 637, 645, 764] # facade
    #st_paul
    #ts_list = [283,317,321,343,348]#facade
    #ts_list = [290,309,343,366,396]#portal,s
    #ts_list = [285,298,328,334,394]#tower,s
    #ts_list = [328,336,346,353,390]#window,s
    #notredame
    #ts_list = [934,1182,1215,1227,1751]#facade
    #ts_list = [1171,1200,1211,1230,1285]#portal,s
    #ts_list = [869,900,1210,1224,1227]#tower,s
    #ts_list = [1192,1203,1211,1227,1239]#window,s
    #blue_mosque
    #ts_list = [354,384,188,153,24,4]#domes
    #ts_list = [15,145,154,157,193]#minaretes
    #ts_list = [8,24,153,354,384]#portals
    ts_list = [4,24,153,354,384]#windows
    #badashahi
    #ts_list = [9,11,347,356,459]#domes
    #ts_list = [11,29,38,178,421]#minarets
    #ts_list = [9,347,356,381,459]#portals
    #ts_list = [9,66,228,356,459]#windows
    #hurba
    #ts_list = [28,37,52,55,58]#domes
    #ts_list = [26,44,45,52,101]#portals
    #ts_list = [28,37,45,52,55]#windows

    categories = ['facade','portal','spires','window']
    gt_idx = []
    gt_files = []
    pred_idx = []
    for dir in categories:
        for file in os.listdir(gt_path):
            if re.findall('[0-9]+',file):
            #if re.findall('[0-9]+',file) and re.findall('mask_gt',file):
                if int(re.findall('\d+',file)[0]) in ts_list :
                    if file.endswith(".jpg"):
                        gt_idx.append(int(re.findall('[0-9]+',file)[0]))
                        gt_files.append(file)

        for file in os.listdir(eval_path):
            if file.endswith(".png"):
                int_idx = int(re.findall('\d+',file)[0])
                if int(re.findall('\d+',file)[0]) in gt_idx:
                    #if re.match(file,'[0-9][0-9][0-9]'):
                    #    images[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
                    #elif re.match(file,'[0-9][0-9][0-9]_s_gt'):
                    #    images_gt[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
                    #elif re.match(file,'[0-9][0-9][0-9]_s_f'):
                    #if re.findall('score_pred',file):
                    scores.append(os.path.join(eval_path, file))
                    pred_idx.append(int(re.findall('[0-9]+',file)[0]))
                    mask_gt.append(os.path.join(gt_path, gt_files[gt_idx.index(int_idx)]))
                    #elif re.match(file,'[0-9][0-9][0-9]_s_gt'):
                    #    scores_gt[re.find('[0-9][0-9][0-9]',file)] = os.path.join(path, file)
                else:
                    continue

        #rmv_idx = list(set(gt_idx) - set(pred_idx))
        #for file in os.listdir(gt_path):
        #    if re.findall('[0-9]+',file):
        #        if int(re.findall('\d+',file)[0]) in rmv_idx:
        #            mask_gt.remove(os.path.join(gt_path, file))

                    

    
    labels = ['portal'] * len(mask_gt)
    #labels = ['apple'] * len(mask_gt)
    calculator = MetricCalculator(eval_path)
    calculator.process_all_images(scores,mask_gt,labels)