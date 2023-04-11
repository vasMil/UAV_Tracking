from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.ssd import SSD, _vgg_extractor
from torchvision.models.vgg import vgg16
# https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from nets.DetectionNetBench import DetectionNetBench
from GlobalConfig import GlobalConfig as config

class Detection_FasterRCNN(DetectionNetBench):
    def __init__(self,
                 root_train_dir: str = "",
                 json_train_labels: str = "",
                 root_test_dir: str = "",
                 json_test_labels: str = ""
            ) -> None:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        super().__init__(model,
                         "FasterRCNN",
                         root_train_dir,
                         json_train_labels,
                         root_test_dir,
                         json_test_labels
                    )
        if not config.profile: self.prof = None

class Detection_SSD(DetectionNetBench):
    def __init__(self,
                 root_train_dir: str = "",
                 json_train_labels: str = "",
                 root_test_dir: str = "",
                 json_test_labels: str = ""
            ) -> None:
        # Customize the model
        backbone = vgg16(weights=None, progress=True)
        backbone = _vgg_extractor(backbone, False, 5)
        
        # Source: https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/torchvision/models/detection/ssd.py#L466
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        
        model = SSD(backbone=backbone,
                         anchor_generator=anchor_generator,
                         size=(300, 300),
                         num_classes=2
                    )
        
        super().__init__(model,
                         "SSD",
                         root_train_dir,
                         json_train_labels,
                         root_test_dir,
                         json_test_labels
                    )
        self.scheduler = None
        if not config.profile: self.prof = None
