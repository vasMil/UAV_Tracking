# UAV_Tracking
The goal of this repository is to develop a pipeline for tracking a Unmanned Aerial Vehicle (UAV) that moves in a 3D space, using another UAV equipped with a single (monocular) camera and some sort of on board unit for processing, like a Jetson device to run a Convolutional Neural Network (CNN).

The code is designed to run using [AirSim](https://github.com/microsoft/AirSim).
Unfortunately it has been discontinued by Microsoft, but it still has a significant value as a project to get started with Computer Vision and Control.

<div align="center">
  <a href="(https://youtu.be/14hfslBnLY0">
    <img src="https://img.youtube.com/vi/14hfslBnLY0/0.jpg" alt="Video Title">
  </a>
</div>

The LeadingUAV is moving on 3 predefined paths that try to combine all sort of possible maneuvers that can be performed. This way, during development we have a good understanding if a change that was made in the pipeline lead to better tracking or not.

## The paths
In the images below you can see the 3 predefined paths that are mentioned in the previous section and are visible in the tracking video above.

1) The first path is a square pulse in two dimensions followed by a spiral that goes up.
2) The second path is a sinusoid in two dimensions but as time passes by it's frequency increases.
3) The third path is a sinusoid in both the y and z axis.

<style>
  div.path-wrap {
    display: flex;
    flex-direction: column;
    max-width: 20em;
    margin: 0 1em;
  }
</style>
<div align="center" style="display:flex; flex-direction:row">
  <div class="path-wrap">
    <img src="images\path_v0.png" alt="path_v0">
    <label>Path v0</label>
  </div>
  <div class="path-wrap">
    <img src="images\path_v1.png" alt="path_v1">
    <label>Path v1</label>
  </div>
  <div class="path-wrap">
    <img src="images\path_v2.png" alt="path_v2">
    <label>Path v2</label>
  </div>
</div>

## The pipeline
The pipeline, depicted in the image below, was designed from scratch. All the modules (except the SSD) are also written from scratch in python which really helped to understand the key points for each and one of them.
<div align="center">
  <img src="images\pipeline.png" alt="Pipeline">
</div>

## The DNN
The SSD was trained on a custom dataset that was generated automatically using AirSim's [object detection](https://microsoft.github.io/AirSim/object_detection/) and [segmentation camera](https://microsoft.github.io/AirSim/image_apis/#segmentation). In order to speedup the process of data generation we utilized the first capability (object detection) that is not as accurate, to get a bounding box. Then, using this bounding box as the Region of Interest (RoI), we scanned all colored pixels returned by the segmentation camera that is ground truth inside this RoI.
This obviously accelerated our implementation, producing 10.000 images with the corresponding bounding boxes in a couple of minutes.
It was also really interesting placing the EgoUAV and LeadingUAV inside the map, with different orientations, trying to always have the LeadingUAV inside EgoUAV's Field Of View.
<div align="center">
  <img src="images\3d_gendata.png" alt="3D camera FOV">
</div>

## Pruning
After getting some measurements for the minimum Frames Per Second (FPS), our DNN needs to operate at, in order to achieve reliable tracking, we realized that we had to find ways to accelerate the inference time.
This is because the on board unit would not be as powerful as a 3090ti is.
A really interesting way to achieve this is through pruning the network.
We chose Structured Pruning over Unstructured Pruning since the first is what truly accelerates the inference.
The way it works is simply by dropping filters based on a criterion, for example the magnitude of the values in the filter.

Following the idea in [Multi-layer Pruning Framework for Compressing Single Shot MultiBox Detector](https://arxiv.org/abs/1811.08342), we divided the SSD layers into sets.
Each set is colored in the image below:
<div align="center">
  <img src="images/ssd_pruning_sets.png" alt="colored SSD Pruning sets">
</div>

In order to avoid the complicated implementation that this paper describes, we decided to perform the sparse training as it has been implemented in the [Torch-Pruning](https://arxiv.org/abs/1811.08342) library and then prune the filters, using the l1-norm criterion.
Last but not least we retrained the pruned network to regain it's accuracy.

Some of the results of pruning and measuring the FPS for the entire pipeline on a Jetson Nano can be found below. The two most interesting points are:
1) We were able to achieve very high levels of sparsity per Set and thus get a significant speedup.
This is a strong indication that the model we started with (SSD300) was really big for what we are trying to achieve.
2) The mAP@75 improves in some cases after the model is pruned. This is a phenomenon that has also been observed in the literature.

<table align="center">
    <tr>
        <td>model_id</td>
        <td>sparsity</td>
        <td>map_75</td>
        <td>ssd_fps</td>
        <td>pipeline_fps</td>
    </tr>
    <tr>
        <td>original</td>
        <td>0.0</td>
        <td>0.9098</td>
        <td>2.8673</td>
        <td>2.8359</td>
    </tr>
    <tr>
        <td>4Sets_80_80_80_80</td>
        <td>0.9688</td>
        <td>0.9662</td>
        <td>14.7077</td>
        <td>12.1137</td>
    </tr>
    <tr>
        <td>4Sets_80_80_90_80</td>
        <td>0.9698</td>
        <td>0.9575</td>
        <td>13.9588</td>
        <td>11.8247</td>
    </tr>
    <tr>
        <td>4Sets_80_80_80_90</td>
        <td>0.9712</td>
        <td>0.9659</td>
        <td>14.5709</td>
        <td>12.3011</td>
    </tr>
    <tr>
        <td>4Sets_85_80_90_85</td>
        <td>0.9754</td>
        <td>0.9547</td>
        <td>15.1691</td>
        <td>12.4295</td>
    </tr>
    <tr>
        <td>4Sets_80_85_80_80</td>
        <td>0.9767</td>
        <td>0.9572</td>
        <td>14.1432</td>
        <td>12.1016</td>
    </tr>
    <tr>
        <td>4Sets_90_80_90_80</td>
        <td>0.9775</td>
        <td>0.9437</td>
        <td>15.3562</td>
        <td>12.9553</td>
    </tr>
    <tr>
        <td>4Sets_90_80_95_80</td>
        <td>0.9779</td>
        <td>0.9181</td>
        <td>15.6067</td>
        <td>12.802</td>
    </tr>
    <tr>
        <td>4Sets_80_85_90_85</td>
        <td>0.9786</td>
        <td>0.9553</td>
        <td>14.0684</td>
        <td>11.76</td>
    </tr>
    <tr>
        <td>4Sets_80_85_80_95</td>
        <td>0.9799</td>
        <td>0.9561</td>
        <td>11.7216</td>
        <td>10.2928</td>
    </tr>
    <tr>
        <td>4Sets_85_85_80_90</td>
        <td>0.9834</td>
        <td>0.9449</td>
        <td>14.8555</td>
        <td>12.5908</td>
    </tr>
    <tr>
        <td>4Sets_90_90_85_90</td>
        <td>0.992</td>
        <td>0.9096</td>
        <td>15.0115</td>
        <td>12.843</td>
    </tr>
    <tr>
        <td>4Sets_90_90_95_90</td>
        <td>0.9923</td>
        <td>0.5526</td>
        <td>14.4126</td>
        <td>11.8041</td>
    </tr>
    <tr>
        <td>4Sets_90_90_90_95</td>
        <td>0.9929</td>
        <td>0.5833</td>
        <td>14.3425</td>
        <td>11.9564</td>
    </tr>
    <tr>
        <td>4Sets_90_95_90_80</td>
        <td>0.9933</td>
        <td>0.4683</td>
        <td>15.6644</td>
        <td>12.7174</td>
    </tr>
    <tr>
        <td>4Sets_90_95_90_85</td>
        <td>0.9944</td>
        <td>0.4682</td>
        <td>14.9561</td>
        <td>12.4952</td>
    </tr>
</table>

In order to compare the original model and the pruned model, we decided to run our tracking application for different velocities, using the `original` model and `4Sets_85_85_80_90` that achieves the best FPS, while maintaining a high accuracy.
The maximum velocities for which the tracking was reliable are summarized in the table below:

<table align="center">
    <tr>
        <td>path\model</td>
        <td>original</td>
        <td>4Sets_85_85_80_90</td>
    </tr>
    <tr>
        <td>path_v0</td>
        <td>0.7 m/s</td>
        <td>6.2 m/s</td>
    </tr>
    <tr>
        <td>path_v1</td>
        <td>0.8 m/s</td>
        <td>5.9 m/s</td>
    </tr>
    <tr>
        <td>path_v2</td>
        <td>1.2 m/s</td>
        <td>5.2 m/s</td>
    </tr>
</table>

It is also important to note that converting the models to a TensorRT engine would accelerate the inference even more, resulting to even higher FPS.

## The Kalman Filter
The formulas for implementing the Kalman Filter can be seen in the figure below.
<div align="center">
  <img src="images\kalman_filter.png" alt="Kalman Filter equations">
</div>


The Kalman Filter requires:
1) The system to be linear.
2) The matrices `F`, `Q`, `H` and `R` to be known.
3) The noise to be white and Gaussian.

Since the LeadingUAV whose movement we are trying to model is performing maneuvers, the true system can not be linear.
The other two requirements also can't be true, but they affect our model a lot less.
In order to correct for these and especially the non-linearity of our system, we chose the `Constant Acceleration` model and followed some of the techniques proposed in the book: "Optimal State Estimation, by Dan Simon".
More specifically we:
1) Forced the `P` matrix to be symmetric.
2) Used a fading-memory filter.
3) Modeled fictitious noise in the Process Noise Covariance Matrix (`Q`).

The last two changes force the system to trust more the measurements than the prediction.

## Vector Transformation
An important concept that has a significant impact on the performance of our tracking algorithm is that we change the yaw of the EgoUAV so that the LeadingUAV is always inside our frame. Since the yaw rotation is independent of our movement *in contrast with pitch and roll*, we are able to turn so we always face the LeadingUAV. This way it is less likely for the LeadingUAV to "escape" on the y axis, thus we are getting more detections and the tracking improves.

At the same time we introduce additional complexity to the problem we are trying to solve. The measurement axis is now different than the movement axis. In order to better understand this a simplified version of this, in 2 dimensions is designed in geogebra and can be found in this link: https://www.geogebra.org/m/rjv6zvk2.

<style>
  .angles-wrap {
    display: flex;
    flex-direction: row;
    align-items: center;
  }
  .angle-wrap {
    margin: 1em;
  }
</style>

<div class="angles-wrap" align="center" style="display:flex; flex-direction:row">
  <div class="angle-wrap">
    <img src="images\yaw.png" alt="yaw">
    <label>Yaw</label>
  </div>
  <div class="angle-wrap">
    <img src="images\pitch.png" alt="pitch">
    <label>Pitch</label>
  </div>
  <div class="angle-wrap">
    <img src="images\roll.png" alt="roll">
    <label>Roll</label>
  </div>
</div>
