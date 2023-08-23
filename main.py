from nets.DetectionNets import Detection_SSD
from pruning import prune_ssd

if __name__ == '__main__':
    ssd = Detection_SSD(root_train_dir="./data/empty_map/train",
                        json_train_labels="./data/empty_map/train/bboxes.json",
                        root_test_dir="./data/empty_map/test",
                        json_test_labels="./data/empty_map/test/bboxes.json",
                        checkpoint_path="./nets/checkpoints/pruning/ssd_pretrained/sparse_training/pretrained60_sparse80.checkpoint"
    )
    
    prune_ssd(ssd=ssd,
              ssd_checkpoint_filename="pretrained60_sparse80.checkpoint",
              sparsities=(0.9, 0.9, 0.9, 0.9,),
              checkpoint_folder="./nets/checkpoints/pruning/ssd_pretrained/finetuning/4layers_09_09_09_09/"
    )
