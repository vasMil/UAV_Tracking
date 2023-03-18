import airsim
import numpy as np
import torch

from models.UAV import UAV
from GlobalConfig import GlobalConfig as config

class EgoUAV(UAV):
    def __init__(self, client: airsim.MultirotorClient, name: str) -> None:
        super().__init__(client, name)
        # Takeoff
        self.lastAction = client.takeoffAsync(vehicle_name=name)


    def _getImage(self) -> torch.Tensor:
        # Respond is of type list[airsim.ImageResponse], 
        # since the request is only for a single image,
        # I may extract only the first element of the list
        resp = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)], 
                vehicle_name=self.name
            )[0]
        # Convert the string to a 1D numpy array (dtype = uint8)
        img = np.fromstring(resp.image_data_uint8, dtype=np.uint8)
        # Convert the numpy array to a pytorch tensor
        img = torch.from_numpy(img)
        # Reshape it into the proper tensor dimensions an rgb image should have
        # HxWx3 (=144x256x3)
        img = img.reshape(resp.height, resp.width, 3)
        return img
    