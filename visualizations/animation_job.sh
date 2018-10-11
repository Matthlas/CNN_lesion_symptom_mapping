#!/bin/bash

#$ -q JM
#$ -cwd
#$ -pe smp 21
#$ -j y
#$ -N lesion_masks_3D_render

source activate theano_py27
python 3d_animation.py
