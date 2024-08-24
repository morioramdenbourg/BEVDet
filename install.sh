conda create --name bevradar python=3.8 -y
conda activate bevradar
# install stable pytorch version: please use this version as there are a lot of dependency conflicts with mmdetection and cuda
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# especially max version 1.6.0
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install openmim
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
pip install lyft_dataset_sdk networkx==2.2 numba==0.53.0 numpy nuscenes-devkit plyfile scikit-image==0.19.3 tensorboard trimesh==2.35.39 --ignore-installed llvmlite==0.36.0

# pip install pycuda

# apt-get update
# apt install libgl1-mesa-glx
# apt-get install libglib2.0-dev
# pip install torchvision==0.11.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/

cd BEVDet
python setup.py develop
pip install opencv_python==4.7.0.72 # 3.4.10.37
pip install numpy==1.23.5
pip install setuptools==59.5.0
python3 -m pip install --user protobuf==3.20.1

pip uninstall openxlab

pip install wandb


# MinkowskiEngine
# sudo apt install build-essential python3-dev libopenblas-dev
# pip install ninja
# pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# DSVT
# pip install torch_scatter==2.0.9
# python mmdet3d/models/dsvt_plugin/setup.py develop # compile `ingroup_inds_op` cuda operation

pip install einops==0.6.1
pip install yapf==0.32.0
pip install timm==0.6.13

pip install spconv-cu113

pip uninstall setuptools
pip install setuptools==59.5.0

pip install hydra-core numba



pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install llvmlite==0.36.0

# aimotive
pip install kornia==0.6.1
pip install albumentations==1.3.1
pip install 'laspy[laszip]'==2.5.3
pip install open3d==0.17.0