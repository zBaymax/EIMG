# Metrics

+ The evaluation metrics can be divided into two categories: those related to the quality of the generated music and those related to the emotion relevance to the original image.

## Usage

+ Music Quality:
  
  1. Input the path of the generated music into the `main` function of the `Music_Quality.py` script.
  
  2. Run `python Music_Quality.py` to obtain the metrics related to the quality of the generated music.
+ Emotion Matching
  1. The `emotionmatch` folder contains the training network used to  predict the VA values of generated music. Place our music dataset under the `emotionmatch\dataset` path and run `train.py` to start the training process.
  
  2. To obtain Emotion Matching metrics, run `va_dis.py` with the trained weights in the previous step and generated music as inputs.