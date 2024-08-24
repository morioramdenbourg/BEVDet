from mmdet3d.registry import MODELS
from mmdet3d.models import Base3DDetector

import time

class OutputStore:

    PREPROCESSED = 'preprocessed'
    GRIDS = 'grids'
    BACKBONES = 'backbones'
    NECK = 'neck'
    HEATMAPS = 'heatmaps'
    PREDICTIONS = 'predictions'

    def __init__(self, enabled):
        self.enabled = enabled
        self.store = {}

    def check_enabled(self):
        if not self.enabled:
            raise Exception('OutputStore is not enabled')

    def put_output(self, name, output):
        self.check_enabled()
        self.store[name] = output

    def get_output(self, name):
        self.check_enabled()

        if name in self.store:
            return self.store[name]
        else:
            raise Exception(f'{name} is not in store')


@MODELS.register_module()
class KPPillarsBEV(Base3DDetector):

    def __init__(self,
                 data_preprocessor,
                 encoder,
                 backbone,
                 neck,
                 bbox_head,
                 store_output,
                 init_cfg,
                 **kwargs):
        super(KPPillarsBEV, self).__init__(data_preprocessor, init_cfg, **kwargs)
        self.encoder = MODELS.build(encoder)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.output_store = OutputStore(store_output)

    def _forward(self, inputs):
        preprocessed = inputs['points']
        grids = self.encoder(inputs)
        backbones = self.backbone(grids)
        neck = self.neck(backbones)

        self.output_store.put_output(OutputStore.PREPROCESSED, preprocessed)
        self.output_store.put_output(OutputStore.GRIDS, grids)
        self.output_store.put_output(OutputStore.BACKBONES, backbones)
        self.output_store.put_output(OutputStore.NECK, neck)

        return neck

    def loss(self, inputs, data_samples):
        start_time = time.time()
        print("IN")
        neck = self._forward(inputs)
        car_features = self.get_car_features(neck)

        bbox_heatmaps = self.bbox_head(car_features)
        bbox_predictions = self.bbox_head.predict(car_features, data_samples)

        self.output_store.put_output(OutputStore.HEATMAPS, bbox_heatmaps)
        self.output_store.put_output(OutputStore.PREDICTIONS, bbox_predictions)

        loss = self.bbox_head.loss(car_features, data_samples)

        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"OUT: {elapsed_time}")
        return loss

    def predict(self, inputs, data_samples):
        start_time = time.time()
        print("IN PREDICT")
        neck = self._forward(inputs)
        car_features = self.get_car_features(neck)
        bbox_predictions = self.bbox_head.predict(car_features, data_samples)

        results_list_2d = None
        detsamples = self.add_pred_to_datasample(data_samples,
                                                 bbox_predictions,
                                                 results_list_2d)
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"OUT PREDICT: {elapsed_time}")
        return detsamples

    def extract_feat(self, inputs, data_samples):
        pass

    @staticmethod
    def get_car_features(neck):
        fpn_car_layer = 1
        car_features = neck[fpn_car_layer:fpn_car_layer + 1]
        return car_features

