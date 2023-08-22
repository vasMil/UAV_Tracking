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
        model = ssd300_vgg16(weights_backbone=None,
                             num_classes=2,
                             trainable_backbone_layers=5)
        config = DefaultTrainingConfig()
        config.default_batch_size = 32
        config.num_workers = 0
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
        self.losses_plot_ylabel = "Sum of localization (Smooth L1 loss) and classification (Softmax loss)"
