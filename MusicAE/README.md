# Music Autoencoder

We use Music FaderNets[1] and LSR [2] as the backbone to extract music features and generate music.

## Usage

1. Download our music dataset.

2. Process the dataset using `process_data.py` to obtain the rhythm, note-density, valence, and arousal information.

3. Modify the training configurations in `model_config_v2.json` / `gmm_model_config.json`.

4. Run `python trainer_gmm.py` / `python trainer_singlevae.py` to start the training.

5. The trained model weights can be found in `params/folder`.

6. If you want to generate new music, run `python generate_gmm.py` / `python generate_single.py`

7. If you want to extract music features for further training, run `python produce_vector_gmm.py` / `python produce_vector_single.py`

8. Given the generated music features, generation can be performed using `python reconstruct_gmm.py` / `python reconstruct_gmm.py`.



[1] H. Tan, and D. Herremans, "Music FaderNets: Controllable Music Generation Based on High-Level Features via Low-Level Feature  Modelling," in *Proc. International Society for Music Information Retrieval Conference*, 2020, 109-116.

[2] A. Pati and A. Lerch, “Latent Space Regularization for Explicit Cntrol of Musical Attributes,” in ICML Machine Learning for Music Discovery Workshop, 2019.