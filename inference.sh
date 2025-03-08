#!/bin/bash

python inference.py                                                  \
    --minicpm_path "/mnt/data/group/models/flux/MiniCPM-o-2_6"       \
    --flux_path    "/mnt/data/group/models/flux/shuttle-3-diffusion" \
    --num_steps 4  --num_gen_imgs 1 --task "all"
