from torch import optim

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.ssd import ssd300_vgg16

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
                 json_test_labels: str = ""
            ) -> None:
        model = ssd300_vgg16(weights=None, num_classes=2)
        super().__init__(model,
                         "SSD",
                         root_train_dir,
                         json_train_labels,
                         root_test_dir,
                         json_test_labels
                    )
        # Overwrite default behaviour of DetectionNetBench
        # Change the learning rate for the optimizer,
        # since the default value results to nan values
        self.optimizer.param_groups[0]['lr'] = 0.0001
        # Update the batch size
        self.batch_size = 32
        # Remove the scheduler
        self.scheduler = None
        if not config.profile: self.prof = None