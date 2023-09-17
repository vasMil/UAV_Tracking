from typing import Optional

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.ssd import ssd300_vgg16, VGG16_Weights

from config import DefaultTrainingConfig
from nets.DetectionNetBench import DetectionNetBench

class Detection_FasterRCNN(DetectionNetBench):
    def __init__(self,
                 config: DefaultTrainingConfig = DefaultTrainingConfig(),
                 checkpoint_path: Optional[str] = None
            ) -> None:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        super().__init__(model=model,
                         model_id="FasterRCNN",
                         config=config,
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
                 use_pretrained_vgg: bool = False,
                 trainable_backbone_layers: int = 5,
                 checkpoint_path: Optional[str] = None
            ) -> None:
        weights_backbone = VGG16_Weights.DEFAULT if use_pretrained_vgg else None
        model = ssd300_vgg16(weights_backbone=weights_backbone,
                             num_classes=2,
                             trainable_backbone_layers=trainable_backbone_layers)
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
                         checkpoint_path=checkpoint_path
                    )
        self.losses_plot_ylabel = "Sum of localization (Smooth L1 loss) and classification (Softmax loss)"
