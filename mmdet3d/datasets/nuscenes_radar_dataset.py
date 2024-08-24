from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesRadarDataset(NuScenesDataset):

    def prepare_data(self, index):
        result = super().prepare_data(index)

        data = self.get_data_info(index)

        # result['data_samples'].set_metainfo({'map': data['map']})
        result['data_samples'].set_metainfo({'lidar2ego_rotation': data['lidar2ego_rotation']})
        result['data_samples'].set_metainfo({'lidar2ego_translation': data['lidar2ego_translation']})
        result['data_samples'].set_metainfo({'ego2global_rotation': data['ego2global_rotation']})
        result['data_samples'].set_metainfo({'ego2global_translation': data['ego2global_translation']})
        result['data_samples'].set_metainfo({'scene_token': data['scene_token']})
        result['data_samples'].set_metainfo({'sample_index': data['sample_index']})
        result['data_samples'].set_metainfo({'sample_token': data['token']})

        # token = data['token']
        # print(f'Retrieving information for sample_data token \'{token}\'')

        return result

