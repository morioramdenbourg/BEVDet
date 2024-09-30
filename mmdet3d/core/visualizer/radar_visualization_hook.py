from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from projects.KPPillarsBEV.kppillarsbev.kppillarsbev import OutputStore
from .radar_visualizer import RadarVisualizer

@HOOKS.register_module()
class RadarVisualizationHook(Hook):

    def __init__(self):
        self.visualizer: RadarVisualizer = RadarVisualizer.get_current_instance()

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch,
                         outputs):

        return

        # return
        batch_num = 0
        if batch_idx != batch_num:
            return

        maps = RadarVisualizationHook.group_maps(data_batch)
        first_key = list(maps.keys())[0] # TODO: Render more maps in the future
        print("map:", first_key)
        record = maps[first_key]
        samples = record['samples']
        points = record['points']
        num_samples = len(samples)
        for index in range(num_samples):
            first_points = points[index]
            first_sample = samples[index]


            # self.visualizer.draw_points_with_map_and_bounding_boxes(first_sample, first_points)

            # preprocessed = runner.model.output_store.get_output(OutputStore.PREPROCESSED)
            # first_points_preprocessed = preprocessed[0].detach()
            RadarVisualizer.draw_points_with_map_and_features_and_bounding_boxes(first_sample, first_points)

            grids = runner.model.output_store.get_output(OutputStore.GRIDS)
            # first_grids = [grid_s[0].detach() for grid_s in grids]
            first_grids = [grids[index].detach()]
            RadarVisualizer.draw_grids_on_map(first_sample, first_grids, first_points, 'Grid')

            backbones = runner.model.output_store.get_output(OutputStore.BACKBONES)
            # first_backbone = [backbone_s[0].detach() for backbone_s in backbones]
            first_backbone = [backbones[index].detach()]
            RadarVisualizer.draw_grids_on_map(first_sample, first_backbone, first_points, 'Backbone')
            #
            # necks = runner.model.output_store.get_output(OutputStore.NECK)
            # first_neck = [neck_s[0].detach() for neck_s in necks]
            # RadarVisualizer.draw_grids_on_map(first_sample, first_neck, first_points, 'FPN')

            predictions = runner.model.output_store.get_output(OutputStore.PREDICTIONS)
            first_predictions = predictions[index].detach()
            RadarVisualizer.draw_predicted_bounding_boxes(first_sample, first_predictions, first_points)

    @staticmethod
    def group_maps(data_batch):
        maps = {}
        for index in range(len(data_batch['data_samples'])):
            sample = data_batch['data_samples'][index]
            points = data_batch['inputs']['points'][index]
            map_file = sample.metainfo['map']['filename']
            if map_file not in maps:
                maps[map_file] = {
                    'points': [points],
                    'samples': [sample]
                }
            else:
                maps[map_file]['points'].append(points)
                maps[map_file]['samples'].append(sample)
        return maps


