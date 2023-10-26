# Implementation of 'Continuous Emotion-Based Image-to-Music Generation'

## Repository organization

+ `ImageAE`: Extract image features.
+ `MusicAE`: Extract music features and generate music.
+ `Cross-Modal Feature Learning`: Transform visual features into music space with the guidance of emotion.
+ `Metrics`: Evaluate our method by analyzing the quality of the generated music and the emotional relevance of the original image.
+ **Note: there is a `README` file included in each path, which contains detailed instructions on how to use the corresponding contents.**

## Usage

1. Use `ImageAE` and `MusicAE` to extract image and music features.

2. Utilize `Cross-Modal Feature Learning` to transform visual features into the music space with the guidance of emotion.

3. Feed the learned music features into `MusicAE` for music generation.

4. Use `Metrics` to evaluate the effectiveness of the method.