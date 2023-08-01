from typing import List, TypeVar, Generic, Optional

import numpy as np

class PepperFilter():
    
    def __init__(self, n_samples: int = 2) -> None:
        self.n = n_samples
        self.samples = RollingQueue[np.ndarray](size=n_samples)
        # Preserve the time that has passed from the last captured measurement saved by the filter
        # so you may later determine the threashold
        self.time_interval = 0

    
    def _filter_measurement(self,
                            meas: Optional[np.ndarray],
                            threashold: float
        ) -> Optional[np.ndarray]:
        if meas is None:
            return None

        prev_meas = self.samples.last()
        sque_meas = meas.squeeze()
        for i, val in enumerate(sque_meas):
            prev_val = prev_meas[i]
            if abs(val - prev_val) > threashold:
                return None
            
        if np.linalg.norm(sque_meas - prev_meas) > threashold:
            return None

        self.samples.push(sque_meas)
        return meas

    def step(self,
             meas: Optional[np.ndarray],
             max_magn_uav_vel: float,
             time_interval: float
        ) -> Optional[np.ndarray]:
        if meas is not None and len(self.samples) == 0:
            self.samples.push(meas)
            return meas

        # Add the time_interval to the time that has passed
        # since the previous value saved by the filter
        self.time_interval += time_interval

        # Determine the threashold
        threashold = 2*max_magn_uav_vel*self.time_interval
        filt_meas = self._filter_measurement(meas, threashold=threashold)

        # If the returned value is not None then a new measurement
        # was saved by the filter, thus we should reset the time_interval
        if filt_meas is not None:
            self.time_interval = 0
        return filt_meas
        


T = TypeVar('T')
class RollingQueue(Generic[T]):
    def __init__(self,
                 size: int,
                 init_val: Optional[T] = None
        ) -> None:
        self.elements: List[T] = []
        self.size = size
        if init_val is not None:
            for _ in range(size):
                self.elements.append(init_val)
    
    def __getitem__(self, item):
        if item > self.size:
            raise Exception("Index exceeds the size of the RollingQueue")
        return self.elements[item]
    
    def __len__(self):
        return len(self.elements)
    
    def push(self, el: T) -> T:
        self.elements.append(el)
        return self.elements.pop(0)
    
    def last(self) -> T:
        return self.elements[self.size-1]
