import feature_utils
import clip_utils
import cv2
import os
import re
import imageio

#blue_mosque
#ts_list = [354,384,188,190,153,24,10,4]#domes
#ts_list = [15,145,154,157,193]#minarets
#ts_list = [8,24,153,190,354,384]#portals
#ts_list = [4,10,24,153,190,354,384]#windows
#badashahi
#ts_list = [9,11,347,356,459]#domes
#ts_list = [11,29,38,178,421]#minarets
#ts_list = [9,347,356,381,459]#portals
#ts_list = [9,66,228,356,459]#windows

def clculate_score(out_dir,dir, file_name, clip_editor,suffix_img):
    pred_img = imageio.imread(os.path.join(dir,file_name))
    w = pred_img.shape[0]
    h = pred_img.shape[1]
    scores = clip_editor.calculate_selection_score(pred_img, query_features=clip_editor.text_features.cuda())
    score_patch = scores.reshape(1, h, w, 1).permute(0, 3, 1, 2).detach()
    score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
    activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
    activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
    id = int(re.findall('\d+',file)[0])
    imageio.imwrite(os.path.join(out_dir, f'{id:03d}_'+suffix_img+'.png'), activation_heatmap)
    imageio.imwrite(os.path.join(out_dir, f'{id:03d}_score_pred.png'), score_pred)
    
dir = "/storage/hanibezalel/Ha-NeRF/eval_badashahi_m_200/results/phototourism/figure_out_dome"
clipnerf_text = "dome"
out_dir = "/storage/hanibezalel/Ha-NeRF/eval_badashahi_m_200/results/phototourism/figure_out_dome"

clip_editor = clip_utils.CLIPEditor()
clip_editor.text_features = clip_editor.encode_text(clipnerf_text)
for file in os.listdir(dir):
    if re.findall('f_f',file):
        clculate_score(out_dir,dir, file, clip_editor,"s_f")
    if re.findall('f_gt',file):
        clculate_score(out_dir,dir, file, clip_editor,"s_gt")
        

