import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.networks import *
from models.nerf import *

from datasets import dataset_dict
from datasets.depth_utils import *
from datasets.colmap_utils import *

from utils import load_ckpt
from datasets.PhototourismDataset import *
import math
from PIL import Image
from torchvision import transforms as T
import cv2

from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

###dff
#import feature_utils
import clip_utils
###dff
MIN_MATCH_COUNT = 10

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/cy/PNW/datasets/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'test_train', 'test_test'])
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='gif',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')
    
    parser.add_argument('--save_dir', type=str, default="./",
                        help='pretrained checkpoint path to load')
    
    ###dff
    parser.add_argument('--feature_dim', type=int, default=None)
    parser.add_argument('--use_semantics', default=False, action="store_true",
                        help='whether to use semantic labels')
    parser.add_argument('--clipnerf_text', nargs='*', type=str, default=None)
    parser.add_argument('--features_path', type=str, default=None)
    
    ###dff

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)

    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R





def define_poses_milan(dataset,dataset_reg,relative_trans=None):
    N_frames = 3
    pose_2 = dataset_reg.poses_dict[483]
    dataset.poses_test = np.tile(pose_2, (N_frames, 1, 1))
    R,T=find_transformation_from_bin(dataset_reg)
    pose_1 = dataset_reg.poses_dict[598]
    #dataset.poses_test[1,:,:]=pose_1
    T_d=np.dot(R,np.expand_dims(dataset.poses_test[1, :, 3],1))
    dataset.poses_test[1, 0, 3] =T[0]+ T_d[0]
    dataset.poses_test[1, 1, 3] =T[0]+ T_d[1]
    dataset.poses_test[1, 2, 3] =T[0]+ T_d[2]
    dataset.poses_test[1, :, :3] = np.dot(R,pose_1[ :, :3])
    dataset.poses_test[2,:,:]=pose_1
    
    

def define_camera(dataset):
    # define testing camera intrinsics (hard-coded, feel free to change)
    dataset.test_img_w, dataset.test_img_h = args.img_wh
    dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/(6)) # fov=50 degrees
    dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                [0, dataset.test_focal, dataset.test_img_h/2],
                                [0,                  0,                    1]])
    
def find_transformation_from_bin(dataset_reg):
    camdata = read_cameras_binary('/storage/chendudai/data/0_1_undistorted/dense/sparse/cameras.bin')
    imdata = read_images_binary('/storage/chendudai/data/0_1_undistorted/dense/sparse/images.bin')
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    #r_ge = R.from_quat(imdata[483][1]).as_matrix()
    t_2 = imdata[23][2].reshape(3, 1)
    #r_im = R.from_quat(imdata[598][1]).as_matrix()
    t_1 = imdata[25][2].reshape(3, 1)
    R_2 = qvec2rotmat(imdata[483][1])
    R_1 = qvec2rotmat(imdata[598][1])
    w2c_1 = np.concatenate([np.concatenate([R_1, t_1], 1), bottom],0)
    w2c_1 = np.linalg.inv(w2c_1)[:3] # (N_images, 3, 4)
    w2c_1[..., 1:3] *= -1
    w2c_2 = np.concatenate([np.concatenate([R_2, t_2], 1), bottom],0)
    w2c_2 = np.linalg.inv(w2c_2)[:3] # (N_images, 3, 4)
    w2c_2[..., 1:3] *= -1
    #return (np.matmul(R_1.T,R_2),(np.dot(R_1.T,(t_2-t_1))))
    R_1 = w2c_1[ :, :3]
    R_2 = w2c_2[ :, :3]
    t_1 = w2c_1[:,3]
    t_2 = w2c_2[:,3]
    #return (np.matmul(R_1.T,R_2),(np.dot(R_1.T,(t_2-t_1))))
    return (np.matmul(R_2,R_1.T),(t_2-np.dot(np.matmul(R_2,R_1.T),t_1)))
    
    
if __name__ == "__main__":
    args = get_opts()
    img_lerf_path = "rgb_image.jpeg"
    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache
        ###dff
        #kwargs['use_semantics'] = args.use_semantics
        #kwargs['feature_dim'] = args.feature_dim
        #kwargs['features_path'] = args.features_path
    dataset = PhototourismDataset()
    dataset_reg = dataset_dict[args.dataset_name](**kwargs)
    #relative_trans = find_transformation_from_bin()

    clip_editor = clip_utils.CLIPEditor()
    
    
    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
 
    # enc_a
    enc_a = E_attr(3, args.N_a).cuda()
    load_ckpt(enc_a, args.ckpt_path, model_name='enc_a')
    
    
    nerf_coarse = NeRF('coarse',
                    in_channels_xyz=6*args.N_emb_xyz+3,
                    in_channels_dir=6*args.N_emb_dir+3,
                    use_semantics=args.use_semantics,
                    out_feature_dim=args.feature_dim).cuda()

    nerf_fine = NeRF('fine',
                    in_channels_xyz=6*args.N_emb_xyz+3,
                    in_channels_dir=6*args.N_emb_dir+3,
                    encode_appearance=args.encode_a,
                    in_channels_a=args.N_a,
                    use_semantics=args.use_semantics,
                    out_feature_dim=args.feature_dim).cuda()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
#
    models = {'coarse': nerf_coarse, 'fine': nerf_fine}
#
    #nerf_coarse_sem = NeRF('coarse',
    #                       enable_semantic=True, num_semantic_classes=2,
    #                        in_channels_xyz=6*args.N_emb_xyz+3,
    #                        in_channels_dir=6*args.N_emb_dir+3,
    #                        is_test=False).cuda()
    #nerf_fine_sem = NeRF('fine',
    #                     enable_semantic=True, num_semantic_classes=2,
    #                     in_channels_xyz=6*args.N_emb_xyz+3,
    #                     in_channels_dir=6*args.N_emb_dir+3,
    #                     encode_appearance=args.encode_a,
    #                     in_channels_a=args.N_a,
    #                     is_test=False).cuda()
    #
    #load_ckpt(nerf_coarse_sem, args.ckpt_path, model_name='nerf_coarse')
    #load_ckpt(nerf_fine_sem, args.ckpt_path, model_name='nerf_fine')
    #models = {'coarse': nerf_coarse_sem, 'fine': nerf_fine_sem}
    
    imgs = []
    #dir_name = os.path.join(args.save_dir, f'render_cam/{args.scene_name}')
    dir_name = './colmap_test'
    os.makedirs(dir_name, exist_ok=True)

    define_camera(dataset)
    define_poses_milan(dataset,dataset_reg)

    kwargs = {}


    imgs = []
    #id_list = [3,21,117,151,149,290,316]
    #org_img = Image.open(os.path.join(args.root_dir, 'dense/images',
    #                                      dataset_reg.image_paths[dataset_reg.img_ids_train[686]])).convert('RGB')
    #imageio.imwrite(os.path.join(dir_name, dataset_reg.image_paths[dataset_reg.img_ids_train[686]]), org_img)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        rays = sample['rays']
        
        #org_img = Image.open(os.path.join(args.root_dir, 'dense/images',
        #                                      dataset_reg.image_paths[dataset_reg.img_ids_train[761]])).convert('RGB')
        org_img = Image.open("/storage/hanibezalel/Ha-NeRF/rgb_image.jpeg").convert('RGB')
        #imageio.imwrite(os.path.join(dir_name, dataset_reg.image_paths[dataset_reg.img_ids_train[761]]), org_img)

        img_downscale = 8
        img_w, img_h = org_img.size
        img_w = img_w//img_downscale
        img_h = img_h//img_downscale
        img = org_img.resize((img_w, img_h), Image.LANCZOS)
        toTensor = T.ToTensor()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = toTensor(img) # (3, h, w)
        whole_img = normalize(img).unsqueeze(0).cuda()
        kwargs['a_embedded_from_img'] = enc_a(whole_img)
        
        results = batched_inference(models, embeddings, rays.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)
        w, h = sample['img_wh']
        #img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        #img_pred_ = (img_pred*255).astype(np.uint8)
        #imageio.imwrite(os.path.join(dir_name, f'rgb_image_dff_mc.png'), img_pred_)
        imageio.imwrite(os.path.join(dir_name, f'rgb_image_'+str(i)+'.png'), results['rgb_fine'].view(h, w, 3))
        #sem_pred = results['semantics_fine'][:,1].view(h, w, 1).cpu().numpy()
        #sem_pred_original = sem_pred
        #sem_pred = (sem_pred * 255).astype(np.uint8)
        #activation_heatmap = cv2.applyColorMap(sem_pred, cv2.COLORMAP_JET)
        #activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
        #imageio.imwrite(os.path.join(dir_name, 'spires_ours.png'), activation_heatmap)
        
        #imageio.imwrite(os.path.join(dir_name, f'rgb_image.png'), img_pred_)
        #visfeat_f_ = feature_utils.feature_vis(results[f'features_fine'],h)
        #imageio.imwrite(os.path.join(dir_name, f'f_f.png'), visfeat_f_)
        #list_args = ["windows","portals","facade","spires","a picture of a cathedral's facade","a picture of a cathedral's portals","a picture of a cathedral's windows","a picture of a cathedral's spires"]
        #for text in list_args:
        #    clip_editor.text_features = clip_editor.encode_text(text)
        #    print(clip_editor.text_features @ clip_editor.text_features.T)
        #    print("shape",clip_editor.text_features.shape)
        #    scores = clip_editor.calculate_selection_score(results[f'features_fine'].cuda(), query_features=clip_editor.text_features.cuda())
        #    score_patch = scores.reshape(1, h, w, 1).permute(0, 3, 1, 2).detach()
        #    score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
        #    activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
        #    activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
        #    imageio.imwrite(os.path.join(dir_name, f's_f_'+text.replace(' ', '_')+'.png'), activation_heatmap)
        #    imageio.imwrite(os.path.join(dir_name, f'score_pred_'+text.replace(' ', '_')+'.png'), score_pred)
        #find_transformation(os.path.join(args.root_dir, 'dense/images',
        #                                  dataset_reg.image_paths[dataset_reg.img_ids_test[76]]),img_lerf_path)
    #imageio.mimsave(os.path.join(dir_name, f'{fig_name}.gif'), imgs, fps=30)