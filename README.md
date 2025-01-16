# SynthText: Generating Emotionally Expressive Talking Faces From Image And Text
SynthText is a neural network system that generates talking face videos from textual input. Text-driven talking face generation is a cutting-edge approach to producing emotionally expressive avatars. This system ensures that facial expressions match the intended emotion while generating realistic and synchronized speech-driven movements.


---
## Dataset
- **Source**: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **Details**:
  - Actors: 91 (48 male, 43 female).
  - Emotions: 6 categories (Anger, Disgust, Fear, Happiness, Neutral, Sadness) with varying intensity levels.
  - Total Clips: 7,442 videos.
  - Resolution: 480x360, 30 FPS.


---
## Installation
Install the required packages
```
pip install -r data_prep/requirements.txt
```

---
## Data Processing 
- Convert Videos to 25 FPS
```
python3 data_prep/convertFPS.py -i VideoFlash/ -o 25fps_video/
```
- Align faces using landmarks
```
python3 data_prep/prepare_data.py -i 25fps_video/ -o hdf5_folder/ --mode 1 --nw 1
```
- Split the data into train & validation
```
python3 script.py -w hdf5_folder/ -t train_hdf5_folder/ -v val_hdf5_folder/ --val_ratio 0.25
```

---
## Model Architecture
**1. Generator**:
  - **Encoders**:
     - **Speech Encoder**: Processes raw text and converts it into speech embeddings.
     - **Image Encoder**: Extracts facial identity features from the reference image.
     - **Emotion Encoder**: Encodes emotion as a low-dimensional embedding.
     - **Noise Encoder**: Adds variability and robustness.
  
  - **Decoder**:
     - Combines embeddings from all encoders to generate frames.

**2. Discriminators**:
   - **Frame Discriminator**: Ensures frame quality.
   - **Emotion Discriminator**: Validates emotion consistency across frames.

### Train the model
First pre-train the emotion discriminator:
```
python3 train.py -i train_hdf5_folder/ -v val_hdf5_folder/ -o ./models/mde/ --pre_train 1 --disc_emo 1 --lr_emo 1e-4
```

Then pre-train the generator:
```
python3 train.py -i train_hdf5_folder/ -v val_hdf5_folder/ -o ./models/pre_gen/ --lr_g 1e-4
```

Finally, train all together:

```
python3 train.py -i train_hdf5_folder/ -v val_hdf5_folder/ -o ./model/tface_emo/ -m ./models/pre_gen/ -mde ./models/mde/ --disc_frame 0.01 --disc_emo 0.001
```

---
## Inference
- Inference from an image and generated speech file:

```
python3 generate.py -im ./data/image_samples/sample01.png -is ./text2speech/all_sen.mp3 -m ./model/ -o results_all/
```
