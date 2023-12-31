import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from models.networks import E_attr
from math import sqrt
import math
import json
from PIL import Image
from torchvision import transforms as T


###dff
import feature_utils
import clip_utils
import cv2
###dff

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
                      rays, ts, N_samples, N_importance, use_disp,
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
                        ts[i:i+chunk] if ts is not None else None,
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

if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache
        ###dff
        kwargs['use_semantics'] = args.use_semantics
        kwargs['feature_dim'] = args.feature_dim
        kwargs['features_path'] = args.features_path
        ###dff
    dataset = dataset_dict[args.dataset_name](**kwargs)
    #dataset_reg.image_paths[dataset_reg.img_ids_train[761]]
    scene = os.path.basename(args.root_dir.strip('/'))

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        # enc_a
        enc_a = E_attr(3, args.N_a).cuda()
        load_ckpt(enc_a, args.ckpt_path, model_name='enc_a')
        kwargs = {}
        if args.dataset_name == 'blender':
            with open(os.path.join(args.root_dir, f"transforms_train.json"), 'r') as f:
                meta_train = json.load(f)
            frame = meta_train['frames'][0]
            image_path = os.path.join(args.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(args.img_wh, Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (4, h, w)
            img = img[:3, :, :]*img[-1:, :, :] + (1-img[-1:, :, :]) # blend A to RGB (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

    #nerf_coarse = NeRF('coarse',
    #                    in_channels_xyz=6*args.N_emb_xyz+3,
    #                    in_channels_dir=6*args.N_emb_dir+3).cuda()
    #nerf_fine = NeRF('fine',
    #                 in_channels_xyz=6*args.N_emb_xyz+3,
    #                 in_channels_dir=6*args.N_emb_dir+3,
    #                 encode_appearance=args.encode_a,
    #                 in_channels_a=args.N_a).cuda()
    #
    ###dff
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

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}
    
    if args.use_semantics:
        clip_editor = clip_utils.CLIPEditor()
        clip_editor.text_features = clip_editor.encode_text(args.clipnerf_text)
        #clip_editor.text_features = clip_editor.encode_text([t.replace('_', ' ') for t in args.clipnerf_text])
        #print([t.replace('_', ' ') for t in args.clipnerf_text])
        #print(clip_editor.text_features @ clip_editor.text_features.T)
        #print("shape",clip_editor.text_features.shape)
    ###dff

    imgs, psnrs, ssims = [], [], []
    ###dff
    f_gt, f_c, f_f = [], [], []
    s_gt, s_c, s_f = [], [], []
    ###dff
    dir_name = os.path.join(args.save_dir, f'results/{args.dataset_name}/{args.scene_name}')
    os.makedirs(dir_name, exist_ok=True)

    # enc_a
    # define testing poses and appearance index for phototourism
    if args.dataset_name == 'phototourism' and args.split == 'test':
        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w, dataset.test_img_h = args.img_wh
        dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                   [0, dataset.test_focal, dataset.test_img_h/2],
                                   [0,                  0,                    1]])
        if scene == 'brandenburg_gate':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[314]])).convert('RGB') # 111 159 178 208 252 314
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = 314 # 85572957_6053497857.jpg
            N_frames = 30*8

            dx1 = np.linspace(-0.25, 0.25, N_frames)
            dx2 = np.linspace(0.25, 0.38, N_frames - N_frames//2)
            dx = np.concatenate((dx1, dx2))

            dy1 = np.linspace(0.05, -0.1, N_frames//2)
            dy2 = np.linspace(-0.1, 0.05, N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0.1, 0.3, N_frames//2)
            dz2 = np.linspace(0.3, 0.1, N_frames - N_frames//2)
            dz = np.concatenate((dz1, dz2))

            theta_x1 = np.linspace(math.pi/30, 0, N_frames//2)
            theta_x2 = np.linspace(0, math.pi/30, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/10, -math.pi/10, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        elif scene == 'trevi_fountain':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[1548]])).convert('RGB') # 10 1336 1548 296 420 1570 1662
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = dataset.img_ids_train[1548] # 85572957_6053497857.jpg
            N_frames = 30*8
            dx = np.linspace(-0.8, 0.7, N_frames)   # + right
            dy1 = np.linspace(-0., 0.05, N_frames//2)    # + down
            dy2 = np.linspace(0.05, -0., N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0.4, 0.1, N_frames//4)  # + foaward
            dz2 = np.linspace(0.1, 0.5, N_frames//4)  # + foaward
            dz3 = np.linspace(0.5, 0.1, N_frames//4)
            dz4 = np.linspace(0.1, 0.4, N_frames - 3*(N_frames//4))
            dz = np.concatenate((dz1, dz2, dz3, dz4))

            theta_x1 = np.linspace(-0, 0, N_frames//2)
            theta_x2 = np.linspace(0, -0, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/6, -math.pi/6, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[dataset.img_ids_train[1548]], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        elif scene == 'sacre_coeur':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[58]])).convert('RGB') # 10 1336 1548 296 420 1570 1662
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = dataset.img_ids_train[58] # 85572957_6053497857.jpg
            N_frames = 30*8
            dx = np.linspace(-2, 2, N_frames)   # + right

            dy1 = np.linspace(-0., 2, N_frames//2)    # + down
            dy2 = np.linspace(2, -0., N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0, -3, N_frames//2)  # + foaward
            dz2 = np.linspace(-3, 0, N_frames - N_frames//2)  # + foaward

            dz = np.concatenate((dz1, dz2))

            theta_x1 = np.linspace(-0, 0, N_frames//2)
            theta_x2 = np.linspace(0, -0, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/6, -math.pi/6, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[dataset.img_ids_train[99]], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        ###dff
        elif scene == 'test_undistorted':

            # select appearance embedding, hard-coded for each scene
            # interpolate between multiple cameras to generate path
            # dataset.poses_test = generate_camera_path(dataset)
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[314]])).convert('RGB')  # 111 159 178 208 252 314
            img_downscale = 4
            img_w, img_h = img.size
            img_w = img_w // img_downscale
            img_h = img_h // img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img)  # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = 314  # 85572957_6053497857.jpg
            # N_frames = 30 * 8
            N_frames = 24

            dx = np.linspace(-0.2, 0.2, N_frames)   # + right
            # dx = np.linspace(0, 1, N_frames)
            # dx = np.linspace(-0.5, 0.6, N_frames)
            # dx = np.linspace(-0., 0, N_frames)


            # dx1 = np.linspace(-0, 0, 2*N_frames//3)
            # dx2 = np.linspace(-0.3, 0.3, N_frames - 2*N_frames//3)
            # dx = np.concatenate((dx1, dx2))


            dy1 = np.linspace(-0., 0, N_frames // 2)  # + down
            dy2 = np.linspace(-0., 0, N_frames - N_frames // 2)
            dy = np.concatenate((dy1, dy2))



            # dy1 = np.linspace(-0, 0, N_frames//3)
            # dy2 = np.linspace(-0.05, 0.05, N_frames//3)
            # dy3 = np.linspace(-0, 0,  N_frames - 2*N_frames // 3)
            # dy = np.concatenate((dy1, dy2,dy3))


            # dz1 = np.linspace(0, 3, N_frames // 2)  # + foaward
            # dz2 = np.linspace(-3, 0, N_frames - N_frames // 2)  # + foaward
            # dz = np.concatenate((dz1, dz2))

            # dz = np.linspace(0, (5/6), N_frames)
            # dz = np.linspace((7 / 11) * (5 / 6), (7 / 11) * (5 / 6), N_frames)
            # dz = np.linspace((6 / 11) * (5 / 6), (6 / 11) * (5 / 6), N_frames)


            # dz1 = np.linspace(-0.1, 0.1, N_frames//3)
            # dz2 = np.linspace(-0, 0, N_frames - N_frames//3)
            # dz = np.concatenate((dz1, dz2))

            dz = np.linspace(0, 0, N_frames)

            # theta_x1 = np.linspace(math.pi / 30, 0, N_frames // 2)
            # theta_x2 = np.linspace(0, math.pi / 30, N_frames - N_frames // 2)
            # theta_x = np.concatenate((theta_x1, theta_x2))
            #
            # theta_y = np.linspace(math.pi / 10, -math.pi / 10, N_frames)

            theta_x = np.linspace(0, 0, N_frames)
            theta_y = np.linspace(0, 0, N_frames)
            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[25], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
            # dataset.poses_test = generate_camera_path(dataset)
            ###dff
        else:
            # pass
            raise NotImplementedError
        
        kwargs['output_transient'] = False
    ###dff
    #milan
    # ts_list = [17, 23, 29, 89, 117, 131, 633] # window,s
    # ts_list = [120, 146, 350, 561, 751, 761, 796] # portal,s
    # ts_list = [156, 259, 415, 455, 473] # spires
    #ts_list = [25, 634, 637, 645, 764] # facade
    #st_paul
    #ts_list = [283,317,321,343,348]#facade
    # ts_list = [290,309,343,366,396]#portal,s
    # ts_list = [285,298,328,334,394]#tower,s
    # ts_list = [328,336,346,353,390]#window,s
    #notredame
    #ts_list = [934,1182,1215,1227,1751]#facade
    # ts_list = [1171,1200,1211,1230,1285]#portal,s
    # ts_list = [869,900,1210,1224,1227]#tower,s
    # ts_list = [1192,1203,1211,1227,1239]#window,s
    #blue_mosque
    ts_list = [354,384,188,190,153,24,10,4]#domes
    # ts_list = [15,145,154,157,193]#minarets
    # ts_list = [8,24,153,190,354,384]#portals
    # ts_list = [4,10,24,153,190,354,384]#windows
    #badashahi
    # ts_list = [9,11,347,356,459]#domes
    # ts_list = [11,29,38,178,421]#minarets
    # ts_list = [9,347,356,381,459]#portals
    #ts_list = [9,66,228,356,459]#windows
    
    #badashahi_new
    
    #ts_list = [7,310,317,406]#domes
    #ts_list = [1]
    #ts_list = [25,33,159,376]#minarets
    #ts_list = [7,310,317,340,406]#portals
    #ts_list = [7,204,317,406]#windows
    
    #blue
    #ts_list = [3,21,137,169,318,345]#domes
    #ts_list = [13,130,138,141,173]#minarets
    # ts_list = [7,21,137,318,45]#portals
    # ts_list = [3,21,137,318,345]#windows
    
    #hurba
    ts_list = [28,37,52,55,58]#domes
    # ts_list = [26,44,45,52,101]#portals
    # ts_list = [28,37,45,52,55]#windows

    # ts_list = []
    # if os.path.exists('/storage/chendudai/data/gt_warps_with_manually_100/milano/windows/'):
    #     list_dir = os.listdir('/storage/chendudai/data/gt_warps_with_manually_100/milano/windows/')
    #     ts_list = [int(f[:4]) for f in list_dir if f.endswith('jpg')]
    # print(ts_list)
    ###dff
    

    for i in tqdm(range(len(dataset))):
    #for i in ts_list:
        ###dff
        sample = dataset[i]
        try:
            ts = sample['ts']
        except:
            continue
        if ts[0] not in ts_list:
            continue
        ###dff
        rays = sample['rays']
        if args.use_semantics:
            feature_gt = sample['feature']

        ###dff

        if (args.split == 'test_train' or args.split == 'test_test') and args.encode_a:
            whole_img = sample['whole_img'].unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
            w, h = sample['img_wh']
            img_GT = np.clip(sample['rgbs'].view(h, w, 3).cpu().numpy(), 0, 1)
            img_GT_ = (img_GT*255).astype(np.uint8)
            
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)

        if args.dataset_name == 'blender':
            w, h = args.img_wh
        else:
            w, h = sample['img_wh']
        
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}.png'), img_pred_)
        imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_gt.png'), img_GT_)
        
        
        ###dff
        if args.use_semantics:
            
            #visfeat_c_ = feature_utils.feature_vis(results[f'features_coarse'],h)
            #imageio.imwrite(os.path.join(dir_name, f'{i:03d}_f_c.png'), visfeat_c_)
            #f_c += [visfeat_c_]
            visfeat_f_ = feature_utils.feature_vis(results[f'features_fine'],h)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_f_f.png'), visfeat_f_)
            f_f += [visfeat_f_]
            visfeat_gt_ = feature_utils.feature_vis(feature_gt,h)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_f_gt.png'), visfeat_gt_)
            f_gt += [visfeat_gt_]
            
            #scores = clip_editor.calculate_selection_score(results[f'features_coarse'], query_features=clip_editor.text_features)
            #score_patch = scores.reshape(1, h, w, 1).permute(0, 3, 1, 2).detach()
            #score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
            #activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
            #activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
            #imageio.imwrite(os.path.join(dir_name, f'{i:03d}_s_c.png'), activation_heatmap)
            #s_c += [activation_heatmap]
            
            scores = clip_editor.calculate_selection_score(results[f'features_fine'].cuda(), query_features=clip_editor.text_features.cuda())
            score_patch = scores.reshape(1, h, w, 1).permute(0, 3, 1, 2).detach()
            score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
            activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
            activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_s_f.png'), activation_heatmap)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_score_pred.png'), score_pred)
            s_f += [activation_heatmap]
            
            scores = clip_editor.calculate_selection_score(feature_gt.cuda(), query_features=clip_editor.text_features.cuda())
            score_patch = scores.reshape(1, h, w, 1).permute(0, 3, 1, 2).detach()
            score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
            score_mask_gt = cv2.threshold(score_pred,240,255,cv2.THRESH_BINARY_INV)[1]
            activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
            activation_heatmap = cv2.cvtColor(activation_heatmap, cv2.COLOR_BGR2RGB)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_s_gt.png'), activation_heatmap)
            imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_mask_gt.png'), score_mask_gt)
            s_gt += [activation_heatmap]
    ###dff
            
            
        
        
        ###dff
        
    if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        imgs, fps=30)
    print('Done')