import os

import pandas as pd

from GlobalConfig import GlobalConfig as config

def remove_entries_of_missing_images(path_to_images: str,
                                     path_to_csv: str
    ) -> None:
    """
    It is quite common to run the data generation script and some frames
    to be empty or just not picked as data. This images are deleted from the folder
    and thus we need a script to also delete any information stored inside the csv
    file that is also written during the generation process.
    """
    df = pd.read_csv(path_to_csv)
    df = df[df["filename"].isin(os.listdir(path_to_images))].reset_index(drop=True)
    df.to_csv(path_to_csv, mode="w", index=False)

def rename_images(path_to_images: str,
                  path_to_csv: str,
                  start_idx: int = 0
    ) -> None:
    """
    Since we delete images and remove entries from the csv file, we would like to have
    the data indexing be uniform and thus have no missing values (i.e. the images to be
    named from (start index).png to ((number of images) + (start index) - 1).png)

    Be careful, filenames should not overlap!!!
    """
    def rename(row: pd.Series):
        old_name = row["filename"]
        if isinstance(row.name, int):
            new_name = str(row.name + start_idx).zfill(config.filename_leading_zeros) + ".png"
        else:
            raise Exception(f"Unexpected index (name) type of row, expecting int, got {type(row.name)}")
        os.rename(os.path.join(path_to_images, old_name),
                  os.path.join(path_to_images, new_name)
        )
        return new_name

    df = pd.read_csv(path_to_csv)
    df["filename"] = df.apply(rename, axis=1)
    df.to_csv(path_to_csv, mode="w", index=False)
