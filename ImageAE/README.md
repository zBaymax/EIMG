# Image Autoencoder

+ We use ALAE [1], VQVAE [2], $$\beta$$-VAE [3] and  as the backbone to extract image features. This section solely presents the processing of ALAE. For VQVAE and $$\beta$$-VAE, there are many repositories available for reference.

## Usage

1. Download our image dataset.

2. Process the dataset using:

   ```
   python prepare_data.py --out ./data/datasets/DATASET_NAME --n_worker N_WORKER RAW_DATASET_PATH
   ```

   Above command should generate two files under the folder `./data/datasets/DATASET_NAME`: `data.lmdb` and `lock.lmdb`.

3. Modify the training configurations in `config.yaml`.

4. Run `python trainer_alae.py` to start the training.

5. The trained model weights can be found at the specified path in the config file.

6. Run `python test_alae.py` to extract image features.



[1] S. Pidhorskyi, D. A. Adjeroh, and G. Doretto, “Adversarial Latent Autoencoders,” in *Proc. IEEE Conference on Computer Vision and Pattern Recognition*, 2020, pp. 14092–14101.

[2] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[3] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework,” in International Conference on Learning Representations, 2017.