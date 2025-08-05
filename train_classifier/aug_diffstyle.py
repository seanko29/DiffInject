import os 
import sys 
import random 
import argparse
import subprocess
import itertools

import shutil
from glob import glob 
import numpy as np
import pandas as pd 

from tqdm import tqdm
from multiprocessing import Pool, Process

# # Pytorch determistic
# random_seed = 1
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))    # 스크립트 경로
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)             # 프로젝트 경로(module import)


def run(args_diffstyle: list): 
            
    """ Augmentation using Diffstyle """
    
    args, content_image_path, style_image_path, gpu_num = args_diffstyle 
    proc_id = os.getpid()
    
    print(os.path.basename(content_image_path), os.path.basename(style_image_path), gpu_num, proc_id)
    
    # create content_dir, style_dir per proc_id  
    args.content_dir = f"./test_images/{args.exp}/{args.pct}/contents_pid_{proc_id}"
    args.style_dir = f"./test_images/{args.exp}/{args.pct}/styles_pid_{proc_id}"  
    os.makedirs(args.content_dir, exist_ok=True)
    os.makedirs(args.style_dir, exist_ok=True)
    shutil.copy(content_image_path, os.path.join(args.content_dir, os.path.basename(content_image_path)))
    shutil.copy(style_image_path, os.path.join(args.style_dir, os.path.basename(style_image_path)))
    script_path = os.path.join(args.root, 'DiffStyle_official-main', 'main.py')
    
    # ----- 기존에 실행한 aug_diffstyle.py resume 시, 이미 추론한 이미지들은 skip
    # content_img_paths = [os.path.join(args.content_dir, f) for f in os.listdir(args.content_dir) if os.path.isfile(os.path.join(args.content_dir, f)) and not os.path.isdir(os.path.join(args.content_dir, f))]
    # style_img_paths = [os.path.join(args.style_dir, f) for f in os.listdir(args.style_dir) if os.path.isfile(os.path.join(args.style_dir, f)) and not os.path.isdir(os.path.join(args.style_dir, f))]
    
    # for content_i, content_lat_pair in enumerate(content_img_paths):       
    #     for style_i, style_lat_pair in enumerate(style_img_paths):
    # # for content_i, content_lat_pair in enumerate(content_lat_pairs):       
    # #     for style_i, style_lat_pair in enumerate(style_lat_pairs):
            
    #         content_path = content_img_paths[content_i]
    #         style_path = style_img_paths[style_i]
            
    #         # print(args)
    #         save_path = 'content_' + content_path.split('/')[-1].split('.')[0] + '_style_' + style_path.split('/')[-1].split('.')[0] + '.png'
    #         if os.path.isfile(os.path.join(args.save_dir, save_path)):
    #             print(f'file exists ... -> {os.path.join(args.save_dir, save_path)}')
                
    #             # remove directory 
    #             if os.path.isdir(args.content_dir):
    #                 shutil.rmtree(args.content_dir)
    #             if os.path.isdir(args.style_dir):
    #                 shutil.rmtree(args.style_dir)
    #             return 
            
    train_cmd = [f'python', script_path, 
                '--diff_style',                       
                '--model_path', args.model_path,                           
                '--content_dir', args.content_dir,                         
                '--style_dir', args.style_dir,                              
                '--save_dir', args.save_dir,                               
                '--config', args.config,                                    
                '--n_gen_step', str(args.n_gen_step),                            
                '--n_inv_step', str(args.n_inv_step),                            
                '--n_test_step', str(1000),                                  
                '--dt_lambda', str(args.dt_lambda),                              
                '--hs_coeff', str(args.h_gamma),                                 
                '--t_noise', str(args.t_boost),                                 
                '--sh_file_name', args.sh_file_name,                        
                '--omega', str(args.omega)] # ,  
                # '--use_mask']   
    # import pdb; pdb.set_trace();                                     
    cmds = [f'export CUDA_VISIBLE_DEVICES={gpu_num}', ' '.join(train_cmd)]
    subprocess.run('; '.join(cmds), shell=True)
    
    # remove directory 
    if os.path.isdir(args.content_dir):
        shutil.rmtree(args.content_dir)
    if os.path.isdir(args.style_dir):
        shutil.rmtree(args.style_dir)
    # remove file 
    for precomputed_styles_pt in glob(os.path.join(PROJECT_DIR, 'precomputed', f'CIFAR10_inv{args.n_inv_step}_styles_*')): # for precomputed_styles_pt in glob(os.path.join(PROJECT_DIR, 'precomputed', f'{args.exp}/{args.pct}', f'FFHQ_inv{args.n_inv_step}_styles_*')): 
        os.remove(precomputed_styles_pt)
    os.remove(os.path.join(args.save_dir, 'grid.png'))
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Bias-Conflict Augmentation using DiffStyle")

    # ---- amplibias
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument("--root", type=str, default=os.path.join(ROOT_PROJECT_DIR), help="Dataset root")
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_DIR, 'results'), help="where the model was saved")

    parser.add_argument("--exp", type=str, default='bffhq', help="Dataset name")     # bar/new-cmnist/bffhq/bar/bar
    parser.add_argument("--data_type", type=str, default='bffhq', help='kind of data used')
    parser.add_argument("--pct", type=str, default="5pct", help="Percent name")
    parser.add_argument("--etc", type=str, default='vanilla', help="Experiment name")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bias_conflict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    
    # ----- diffstyle 
    parser.add_argument("--sh_file_name", type=str, default=os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main', "script_diffstyle_ffhq.sh"))
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main', 'checkpoint/ffhq_p2.pt'))
    parser.add_argument("--config", type=str, default=os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main', "configs/ffhq.yml"))     # Other option: afhq.yml celeba.yml metfaces.yml ffhq.yml lsun_bedroom.yml ...
    parser.add_argument("--save_dir", type=str, default="./results/diffstyle/bffhq/0.5pct/ffhq_p2_inv1000_gamma07")   # output directory
    parser.add_argument("--content_dir", type=str, default="./test_images/bffhq/0.5pct/contents")
    parser.add_argument("--style_dir", type=str, default="./test_images/bffhq/0.5pct/styles")
    parser.add_argument("--h_gamma", type=float, default=0.7)           # 0.3, 0.5, 0.6, 0.7
    parser.add_argument("--dt_lambda", type=float, default=0.9985)     # 1.0 for out-of-domain style transfer.
    parser.add_argument("--t_boost", type=int, default=200)          # 0 for out-of-domain style transfer.

    parser.add_argument("--n_gen_step", type=int, default=1000)
    parser.add_argument("--n_inv_step", type=int, default=1000) # 50 # 1000
    parser.add_argument("--omega", type=float, default=0.0)

    args = parser.parse_args()
    
    args.exp = 'cifar10c' # 'bffhq' 
    args.pct = '1pct' # '0.5pct' '1pct' '2pct' '5pct'
    args.etc = 'vanilla'
    args.bias_conflict_ratio = 0.1 # 0.2, 0.4, 0.6, 0.8, 1.0 
    
    args.sh_file_name = os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main', 'script_diffstyle_cifar10.sh')
    args.model_path = os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main', 'checkpoint/cifar10_p2_res_blocks_1_ema_0.9999_500000.pt')
    args.config = os.path.join(ROOT_PROJECT_DIR, 'DiffStyle_official-main' ,'configs/cifar10.yml')
    
    # number of gpu / cpu processors 
    args.gpu = [1, 2, 3] # [0], [0, 2, 3] : list 
    # args.n_pool = 1 # 20 # 10 : int

    args.save_dir = f'./results/diffstyle/{args.exp}/{args.pct}/cifar10_p2_res_blocks_1_ema_0.9999_500000.pt/gamma03_nomask' # f'./results/diffstyle/{args.exp}/{args.pct}/ffhq_p2_inv1000_gamma07'
    args.content_dir = f'./test_images/{args.exp}/{args.pct}/contents'
    args.style_dir = f'./test_images/{args.exp}/{args.pct}/styles'

    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----- load data 
    top_k_samples_root_dir = os.path.join(args.save_root, 'top_k_samples', f'{args.exp}-{args.pct}-{args.etc}/')
    top_k_samples_data_dir_list = list(itertools.chain(*[glob(os.path.join(top_k_samples_root_dir, label, '*.png')) for label in os.listdir(top_k_samples_root_dir)]))
    bias_aligned_data_dir_list = [y for x in os.walk(os.path.join(args.root, 'data', args.exp, args.pct, 'align')) for y in glob(os.path.join(x[0], '*/*.png'))]
    #! (bias-align, top_k_samples) -> (원본데이터 전체, top_k_samples) 로 수정
    # # bias_conflict_data_dir_list = [y for x in os.walk(os.path.join(args.root, 'data', args.exp, args.pct, 'conflict')) for y in glob(os.path.join(x[0], '*/*.png'))]
    # total_data_dir_list = bias_aligned_data_dir_list + bias_conflict_data_dir_list
    
    num_aug_diffstyle = int(len(bias_aligned_data_dir_list) * args.bias_conflict_ratio)  
    print(f'Number of generated samples needed: {num_aug_diffstyle}')
    content_image_path_list = np.random.choice(top_k_samples_data_dir_list, num_aug_diffstyle)
    style_image_path_list = np.random.choice(bias_aligned_data_dir_list, num_aug_diffstyle)
    gpu_num_list = args.gpu * int(num_aug_diffstyle//len(args.gpu)) + args.gpu[:num_aug_diffstyle%len(args.gpu)] 
    # gpu_num_list = [0]*num_aug_diffstyle # [0,2,3] * int(num_aug_diffstyle//3) + [0,2,3][:num_aug_diffstyle%3]
 
    # save meta.csv 
    df = pd.DataFrame({'content_image_path': content_image_path_list, 'style_image_path': style_image_path_list, 'gpu_num': gpu_num_list})
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir, 'meta.csv'), index=False)
    
    # 돌리고 싶은 process 개수만큼만 args_diffstyle 할당해줘야 됨 
    n_pool = 3 # 20 # 10 : int
    args_diffstyle_list = list(zip([args]*num_aug_diffstyle, content_image_path_list, style_image_path_list, gpu_num_list))
    for i in range(int(num_aug_diffstyle//n_pool)+1):
        print(f"Processing index {n_pool*i}-{n_pool*(i+1)-1} out of {num_aug_diffstyle}...") 
        args_diffstyle = args_diffstyle_list[n_pool*i: n_pool*(i+1)]
        pool = Pool(n_pool)

        procs = []    
        for args in args_diffstyle:
            proc = Process(target=run, args=(args,))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

        # with pool as p: 
        #     res = list(tqdm(p.imap(run, args_diffstyle), total=len(args_diffstyle)))
