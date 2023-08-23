from pprint import pprint

from nets.DetectionNets import Detection_SSD
from pruning import get_model_stats

if __name__ == '__main__':
    ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                        json_train_labels="./data/empty_map/train/bboxes.json",
                        root_test_dir="./data/empty_map/test",
                        json_test_labels="./data/empty_map/test/bboxes.json",
                        checkpoint_path="./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint"
    )
    
    pprint(get_model_stats(ssd))
