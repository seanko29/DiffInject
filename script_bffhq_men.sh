#!/bin/bash

sh_file_name="script_diffstyle.sh"
gpu=0

config="ffhq.yml"     # Other option: afhq.yml celeba.yml metfaces.yml ffhq.yml lsun_bedroom.yml ...
style_dir="./test_images/bffhq_men/contents_old"
content_dir="./test_images/bffhq_men/styles_young"
dt_lambda=0.9985      # 1.0 for out-of-domain style transfer.
t_boost=200           # 0 for out-of-domain style transfer.
n_gen_step=1000
n_inv_step=1000
omega=0.0

# Run for each h_gamma value using a simple for loop
for h_gamma in 0.1 0.3 0.5 0.7 0.9; do
    # Create a unique save directory for each h_gamma value
    save_dir="./bin/bffhq_men/h_gamma_${h_gamma}"
    
    echo "Running with h_gamma = ${h_gamma}, saving to ${save_dir}"
    
    CUDA_VISIBLE_DEVICES=$gpu python main.py --diff_style                       \
                            --content_dir $content_dir                          \
                            --style_dir $style_dir                              \
                            --save_dir $save_dir                                \
                            --config $config                                    \
                            --n_gen_step $n_gen_step                            \
                            --n_inv_step $n_inv_step                            \
                            --n_test_step 1000                                  \
                            --dt_lambda $dt_lambda                              \
                            --hs_coeff $h_gamma                                 \
                            --t_noise $t_boost                                  \
                            --sh_file_name $sh_file_name                        \
                            --user_defined_t_edit 500                           \
                            --omega $omega
                            # --use_mask
    
    echo "Completed run with h_gamma = ${h_gamma}"
done

echo "All runs completed!"
