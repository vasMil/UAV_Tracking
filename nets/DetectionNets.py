from typing import Optional

from torch import optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.ssd import ssd300_vgg16

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
                 config: DefaultTrainingConfig = DefaultTrainingConfig(),
                 root_train_dir: str = "",
                 json_train_labels: str = "",
                 root_test_dir: str = "",
                 json_test_labels: str = "",
                 checkpoint_path: Optional[str] = None
            ) -> None:
        model = ssd300_vgg16(weights=None, num_classes=2)
        super().__init__(model=model,
                         model_id="SSD",
                         config=config,
                         root_train_dir=root_train_dir,
                         json_train_labels=json_train_labels,
                         root_test_dir=root_test_dir,
                         json_test_labels=json_test_labels,
                         checkpoint_path=checkpoint_path
                    )
        # Overwrite default behaviour of DetectionNetBench
        # Change the learning rate for the optimizer,
        # since the default value results to nan values
        self.optimizer.param_groups[0]['lr'] = 0.001
        # Update the batch size
        self.batch_size = 32
        # Remove the scheduler
        self.scheduler = None
        if not config.profile: self.prof = None    
