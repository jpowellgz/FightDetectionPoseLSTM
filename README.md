# Fight Detection using Keypoints, Angles and Bi-LSTM
Fight detection using Open Pose and Bi-LSTM  
Thesis project with paper on review stages for WEA 2026 and participation in Eureka ENC 2024.


## Abstract  
Currently, in the area of computer vision, researchers work on the automatic detection of violent,
dangerous and suspicious situations. Fights between people are among these situations, and in this
work, the aim is to identify them automatically. We present a proposal of a technique for the
classification of videos containing fights. The method is based in the combination of a deep neural
network for human detection known as Open Pose, and the recurrent neural network known as Long
Short Term Memory. Open Pose estimates postures, which we used to calculate vectors that describe
the general movements of people in a video. These vectors are processed through Long Short Term
Memory to classify fights. This method was verified on three labeled datasets, commonly found in
fight detection research: Movie Fight Dataset, Surveillance Camera Fight Dataset and Violence
Detection Dataset. Respectively, we obtained classification accuracy of: 95%, 67% and 81%. The
method we proposed has a similar performance to recent works on fight detection, and it opens
different possibilities for future work, to improve the accuracy of classification and execution times.  
</details>

## Changelog
- **v0.2.0** - 06/2026. Updating and refactoring with uv and Python 3.10.12.
- **v0.1.0.** - 05/2021. First implementation on Python 3.7. 
  -  Open Pose 1.6.0 (CPU Release)  
  - Open CV 4.5.1.48  
  - Tensorflow 2.4  
  - Scikit-Learn 0.24.0  

## Requirements
- [OpenPose models: COCO pose models: pose_iter_440000.caffemodel, pose_deploy_linevec.prototxt](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master)
- [Astral uv](https://docs.astral.sh/uv/getting-started/)

**Disclaimer**: This project has only been tested with CPU. GPU might require different additional steps to setup OpenPose and Tensorflow.

## Instructions
### Setup
Run <code>uv sync --frozen --extra all</code>

### Image extraction
To extract a frame dataset from a fight video dataset use the <code>extract_frames.py</code> script, with the configuration in <code>configs/extraction_config.json</code> adjusted to your dataset and number of frames per video to extract. The script expects a path to the dataset with one subdirectory for fight videos and another for non-fight videos:  

```
dataset  
├── fight
│   ├── video_000.mp4
│   ├── video_001.mp4
│   └── ....
└── no_fight
    ├── video_000.mp4
    ├── video_001.mp4
    └── ....
```

Use the command

<code>uv run python3 extract_frames.py configs/extraction_config.json</code>

### Training

To train on the frame dataset use the <code>train_openpose_lstm.py</code> script with the corresponding config in <code>configs/train_config.json</code>. Adjust the directory parameters to the dataset.

**Parameters**
- directory_config: The location of the frames and directory structure.
- training_config: Choose whether to save the generated vectors from the skeleton angles, or load from a previous run.
- keypoint_model: The parameters for the OpenPose model.
- angle_calculator: The parameters for the skeleton angles calculator, where:
  - angle_bins: Number of fixed angles to lookup.
  - fight_pairs_indexes: Indexes of the COCO pairs to choose for angle calculation. Default is the main COCO keypoint pairs excluding face keypoints (Pairs 1 to 13)
- classification_model: Parameters for the LSTM model. Where:
  - sequence_length: has to be equal to the number of frames extracted per video.
  - vector_size: Vector size given by the angle calculator. By default it will be 260, being angle_bins (20)\*number of limbs (13)

Run the command:

<code>uv python3 train_openpose_lstm.py configs/train_config.json</code>


## Using custom models

To use custom models for keypoints or classification, It's possible to inherit from the model classes in <code>model_base.py</code> to create classes similar to the ones in models directory. Then copying or modifying the <code>train_openpose_lstm.py</code> init models method to use the custom classes. As long as the inputs and outputs match the expected types and shapes, the training scripts will run the models.

## Tested on
Movie Fights Dataset  
Nievas, E. B., Suarez, O. D., García, G. B., & Sukthankar, R. (2011) Movie Fight Detection
Dataset. Recovered from: http://visilab.etsii.uclm.es/personas/oscar/FightDetection/

Violence Detection Dataset  
Aktı, Ş., Tataroğlu, G.A., Ekenel, H.K. Surveillance Camera Fight Dataset. Recovered from:
https://github.com/sayibet/fight-detection-surv-dataset. Access date: May 11 2021

Surveillance Camera Dataset  
Bianculli, M., Falcionelli, N., Sernani, P., Tomassini, S., Contardo, P., Lombardi,M., Dragoni,
A.F. A dataset for automatic violence detection in videos, Data in Brief 33 (2020).
doi:10.1016/j.dib.2020.106587.

