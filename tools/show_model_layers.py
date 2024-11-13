import torch

model = torch.load('work_dirs/models/bevdet_kppillarsbev__single_scale.pth')
state_dict = model['state_dict']

for k, v in state_dict.items():
    if k.startswith('pts_voxel_encoders'):
        k = k[7:]