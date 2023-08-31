import imageio
import torch
from PIL import Image
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('module://backend_interagg')
# from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import cv2
import imageio
from PIL import Image
#import matplotlib
import matplotlib.pyplot as plt


# path = '/storage/chendudai/repos/Ha-NeRF/sem_results/notre_dame_ours_clipsegFtCrops_maxIter12500_top50And50_getOccRetHorizAndClipFtRetHoriz/results/phototourism/for_metric/top_50_nEpochs10/towers_ds2/900_semantic.png'
# path_img = '/storage/chendudai/data/notre_dame_front_facade/dense/images/0900.jpg'

#num = '140'
#path = '/net/projects/ranalab/itailang/chen/repos/Ha-NeRF/save/results/phototourism/st_paul_colonnade_zeroFreq_baseline/'
#path_gt_image = '/net/projects/ranalab/itailang/chen/data/st_paul/dense/images/' + '0' + num + '.jpg'
#path_img = '/storage/hanibezalel/Ha-NeRF/eval_chen/rgb_image_ours.png'
#path_pred = '/storage/hanibezalel/Ha-NeRF/eval_chen/portals_ours.png'
#path_img = '/storage/hanibezalel/Ha-NeRF/eval_dff/rendered_114.png'
#path_pred = '/storage/hanibezalel/Ha-NeRF/eval_dff/rendered_114_spires_s.png'
#path_img = '/storage/hanibezalel/Ha-NeRF/eval_dff_mc/rgb_image_dff_mc.png'
#path_pred = '/storage/hanibezalel/Ha-NeRF/eval_dff_mc/s_f_spires.png'
path_img = '/storage/hanibezalel/Ha-NeRF/eval_lerf/rgb_image.jpeg'
path_pred = '/storage/hanibezalel/Ha-NeRF/eval_lerf/portals.jpeg'
#path_img = '/storage/hanibezalel/Ha-NeRF/eval_dff/rendered_114.png'
#path_pred = '/storage/hanibezalel/Ha-NeRF/eval_gt/spires.png'
pred = cv2.imread(path_pred)
#pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
(height, width) = pred.shape[:2]
#from matplotlib import colormaps # colormaps['jet'], colormaps['turbo']
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import _jet_data
import matplotlib.colors as colors
import matplotlib.cm as cmx

#jet = LinearSegmentedColormap("jet", _jet_data, N=2**8)
#jet_reversed = jet.reversed()
#grey = jet_reversed(pred)
input = "gray"
if input=="jet":
    jet = plt.cm.jet
    jet._init()
    lut = jet._lut[..., :3]

    norm = cv2.NORM_L2
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fm = cv2.FlannBasedMatcher(index_params, search_params)
    # JET, BGR order, excluding special palette values (>= 256)
    fm.add(255 * np.float32([lut[:, (2,1,0)]])) # jet
    fm.train()

    # look up all pixels
    query = pred.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(query)
    output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    pred = np.where(output < 256, output, 0).astype(np.uint8)
    #pred = np.where(output > 125, output, 0).astype(np.uint8)
    #dist = np.uint8([m.distance for m in matches]).reshape(height, width)
elif input=="gray":
    pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
    pred = cv2.bitwise_not(pred)
else:
    "problematic input"
imageio.imwrite('/storage/hanibezalel/Ha-NeRF/eval_lerf/grey_pred_image_ours.png', pred)

img = cv2.imread(path_img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#gt_img = cv2.imread(path_gt_image)
#gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
#gt_img = cv2.resize(gt_img, dsize=(gt_img.shape[1]//2, gt_img.shape[0]//2), interpolation=cv2.INTER_CUBIC)

# plt.imshow(pred)
# plt.show()
# plt.imshow(img)
# plt.show()
# plt.imshow(gt_img)
# plt.show()


# First method for vis
pred = np.expand_dims(pred, axis=2)
#overlay_yellow = np.concatenate([pred,pred,np.zeros_like(pred)], axis=2)    # yellow
#overlay_green = np.concatenate([np.zeros_like(pred),pred,np.zeros_like(pred)], axis=2)   # green
#alpha = 0.5
#result_green = cv2.addWeighted(img, 1 - alpha,overlay_green, alpha, 0)
#result_yellow = cv2.addWeighted(img, 1 - alpha,overlay_yellow, alpha, 0)


#result_green_with_rgb = cv2.addWeighted(img, 1 - alpha,overlay_green, alpha, 0)
#result_yellow_with_rgb = cv2.addWeighted(gt_img, 1 - alpha,overlay_yellow, alpha, 0)

#imageio.imwrite('/storage/hanibezalel/Ha-NeRF/eval_chen/spires_ours_overlay_green.png', result_green_with_rgb)
#imageio.imwrite(os.path.join(path, f'{num}_new_vis_green.png'), result_green)
#imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow.png'), result_yellow)
#
#imageio.imwrite(os.path.join(path, f'{num}_new_vis_green_with_rgb_gt.png'), result_green_with_rgb)
#imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow_with_rgb_gt.png'), result_yellow_with_rgb)
#imageio.imwrite(os.path.join(path, f'{num}_rgb_gt.png'), gt_img)


# Second method for vis
pred_norm = pred / 255
colormap = plt.get_cmap('jet')
heatmap = colormap(pred).squeeze()
h = heatmap[:, :, :3]
#pred_with_overlay = 0.5 * np.multiply(1 - prezd_norm, img/255) + 0.5 * np.multiply(pred_norm, h)
img_ = Image.fromarray(img,"RGB").copy()
h_ = Image.fromarray(h,"RGB").copy()
img_.putalpha(150)
h_.paste(img_,(0,0),img_)
# plt.imshow(pred_with_overlay)
# plt.show()
imageio.imwrite('/storage/hanibezalel/Ha-NeRF/eval_lerf/portals_new.png', h)
#print('saved in:')
#print(path)