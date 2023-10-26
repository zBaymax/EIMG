# Cross-Modal Feature Learning

+ A feature learning mechanism that transform visual features music space with the guidance of emotion.

## Usage

1. Please place the music features, image features into the dataset folder. **Note that for ease of use, the data can be downloaded from the following link. After download, place the data into `dataset/` for use. You can also use the previously described steps to  generate your own data.**

2. Data: https://drive.google.com/file/d/1AKtuVM-rEflMwq8HmTyDL2oN_6EB2RfP/view?usp=sharing

3. Modify the training configurations in `config.py`.

4. Run `python train.py` to start the training.

5. Run `python val.py` to get music features. Note that the obtained music features need to be further fed into the music decoder for music generation. The code for this part is located in the `MusicAE` directory.
