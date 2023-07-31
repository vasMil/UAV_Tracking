from typing import Optional

from torch import optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models import VGG16_Weights

from config import DefaultTrainingConfig
from nets.DetectionNetBench import DetectionNetBench

class Detection_FasterRCNN(DetectionNetBench):
    def __init__(self,
                 config: DefaultTrainingConfig = DefaultTrainingConfig(),
                 root_train_dir: str = "",
                 json_train_labels: str = "",
                 root_test_dir: str = "",
                 json_test_labels: str = "",
                 checkpoint_path: Optional[str] = None
            ) -> None:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        super().__init__(model=model,
                         model_id="FasterRCNN",
                         config=config,
                         root_train_dir=root_train_dir,
                         json_train_labels=json_train_labels,
                         root_test_dir=root_test_dir,
                         json_test_labels=json_test_labels,
                         checkpoint_path=checkpoint_path
                    )
        # Overwrite default behaviour of DetectionNetBench
        # Update the batch size
        self.batch_size = 4
        # Remove weight decay from the optimizer
        self.optimizer.param_groups[0]["weight_decay"] = 0

        if not config.profile: self.prof = None

class Detection_SSD(DetectionNetBench):
    def __init__(self,
                 root_train_dir: str = "",
                 json_train_labels: str = "",
                 root_test_dir: str = "",
                 json_test_labels: str = "",
                 checkpoint_path: Optional[str] = None
            ) -> None:
        defaults = {
            # Rescale the input in a way compatible to the backbone
            "image_mean": [0.48235, 0.45882, 0.40784],
            "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
        }
        model = ssd300_vgg16(weights_backbone=VGG16_Weights.DEFAULT,
                             num_classes=2,
                             **defaults)
        config = DefaultTrainingConfig()
        config.default_batch_size = 4
        config.num_workers = 16
        config.sgd_learning_rate = 0.0001
        config.scheduler_milestones = []
        config.scheduler_gamma = 1
        config.sgd_momentum = 0.9
        config.sgd_weight_decay = 0.0005
        config.profile = False
        super().__init__(model=model,
                         model_id="SSD",
                         config=config,
                         root_train_dir=root_train_dir,
                         json_train_labels=json_train_labels,
                         root_test_dir=root_test_dir,
                         json_test_labels=json_test_labels,
                         checkpoint_path=checkpoint_path
                    )
