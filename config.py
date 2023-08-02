from typing import List, Tuple, Dict, Any

import json
import airsim

from project_types import Filter_t, Motion_model_t, Gimbal_t

def get_gimbal_config() -> Gimbal_t:
    client = airsim.MultirotorClient()
    settings_str = client.getSettingsString()
    if settings_str is None:
        raise Exception("Unable to recover the settings!")
    settings: Dict[Any, Any] = json.loads(settings_str)
    gimbal: Gimbal_t = {"pitch": False, "roll": False, "yaw": False}
    if not ("CameraDefaults" in settings):
        return gimbal
    settings = settings["CameraDefaults"]
    if not ("Gimbal" in settings):
        return gimbal
    settings = settings["Gimbal"]
    if (not ("Stabilization" in settings)):
        Warning("No Stabilation found in AirSim settings, no gibal will be used!")
        return gimbal
    for key in ["Stabilization", "Pitch", "Roll", "Yaw"]:
        if not key in settings:
            continue
        if settings[key] != 0 and settings[key] != 1:
            Warning(f"Only values {{0,1}} are supported for {key}, any non-zero value will be treated as 1!")
        if key != "Stabilization" and settings[key] != 0:
            gimbal[key.lower()] = True
    return gimbal


class DefaultTrainingConfig():
    def __init__(self,
                 num_epochs: int = 25,
                 default_batch_size: int = 4,
                 num_workers: int = 0,
                 sgd_learning_rate: float = 0.001,
                 sgd_momentum: float = 0.9,
                 sgd_weight_decay: float = 0.0005,
                 scheduler_milestones: List[int] = [100],
                 scheduler_gamma: float = 0.1,
                 profile: bool = False,
                 prof_wait: int = 1,
                 prof_warmup: int = 3,
                 prof_active: int = 3,
                 prof_repeat: int = 1,
                 losses_plot_ylabel: str = "Sum of model's losses"
            ) -> None:
        self.num_epochs = num_epochs
        self.default_batch_size = default_batch_size
        self.num_workers = num_workers
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.sgd_weight_decay = sgd_weight_decay
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_gamma = scheduler_gamma
        self.profile = profile
        self.prof_wait = prof_wait
        self.prof_warmup = prof_warmup
        self.prof_active = prof_active
        self.prof_repeat = prof_repeat
        self.losses_plot_ylabel = losses_plot_ylabel


class DefaultCoSimulatorConfig():
    def __init__(self,
                 uav_velocity: float = 5.,
                 score_threshold: float = 0.,
                 max_vel: Tuple[float, float, float] = (5., 5., 5.),
                 min_vel: Tuple[float, float, float] = (1., -5., -5.),
                 weight_vel: Tuple[float, float, float] = (1., 1., 4.),
                 sim_fps: int = 60,
                 simulation_time_s: int = 240,
                 camera_fps: int = 30,
                 infer_freq_Hz: int = 30,
                 filter_freq_Hz: int = 30,
                 filter_type: Filter_t = "KF",
                 motion_model: Motion_model_t = "CA",
                 use_pepper_filter: bool = True,
                 leadingUAV_update_vel_interval_s: int = 4,
                 max_time_lead_is_lost_s: int = 2,
                 max_allowed_uav_distance_m: int = 15
            ) -> None:
        # The magnitude of the velocity vector (in 3D space)
        self.uav_velocity = uav_velocity
        
        # The minimum score, for which a detection is considered
        # valid and thus is translated to EgoUAV movement.
        self.score_threshold = score_threshold

        # The upper an lower limit for the velocity on each axis of both UAVs
        self.max_vel = max_vel
        self.min_vel = min_vel
        
        # Controller constants - converting bbox to velocity
        # Weights are added for y and z coords. This is helpful since the
        # UAV is not going to reach the target position, using a constant
        # velocity in this small time interval (1/inference_freq_Hz).
        # The most important aspect of the tracking is to preserve the
        # LeadingUAV inside your FOV. Thus we require, at the end
        # of each command, to have the LeadingUAV as close to the center
        # of our FOV as possible. In order to achieve this we need to allocate
        # most of velocity's magnitude towards the z axis. The same applies for
        # the y axis, but this weight may be less, since we also take advantage
        # of the yaw_mode, in order to rotate our camera and look at all times
        # at the LeadingUAV.
        self.weight_vel = weight_vel

        self.sim_fps = sim_fps
        self.simulation_time_s = simulation_time_s
        self.camera_fps = camera_fps
        self.infer_freq_Hz = infer_freq_Hz
        self.filter_freq_Hz = filter_freq_Hz
        self.filter_type: Filter_t = filter_type
        self.motion_model: Motion_model_t = motion_model
        self.use_pepper_filter = use_pepper_filter
        self.leadingUAV_update_vel_interval_s = leadingUAV_update_vel_interval_s
        self.max_time_lead_is_lost_s = max_time_lead_is_lost_s
        self.max_allowed_uav_distance_m = max_allowed_uav_distance_m
        self.use_gimbal: Gimbal_t = get_gimbal_config()

        # The simulation fps should be a common multiple
        # of all actions that need to be performed at a specific frequency
        assert(sim_fps/camera_fps == round(sim_fps/camera_fps, 0))
        assert(sim_fps/infer_freq_Hz == round(sim_fps/infer_freq_Hz, 0))
        assert(sim_fps/filter_freq_Hz == round(sim_fps/filter_freq_Hz, 0))

        # Since there are frames exported with information on them,
        # the camera_fps should be a multiple of the inference frequency.
        # Otherwise there will be a simulation frame (i.e. captured at sim_fps frequency)
        # that needs to be evaluated but it won't have been captured.
        assert(camera_fps/infer_freq_Hz == round(camera_fps/infer_freq_Hz))
