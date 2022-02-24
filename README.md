# EditGAN 

<div align="center">
Official code and tool release for: 


**EditGAN: High-Precision Semantic Image Editing**

[Huan Ling](http://www.cs.toronto.edu/~linghuan/)\*, [Karsten Kreis](https://karstenkreis.github.io/)\*,  [Daiqing Li](https://scholar.google.ca/citations?user=8q2ISMIAAAAJ&hl=en), [Seung Wook Kim](https://seung-kim.github.io/seungkim/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)

(\* authors contributed equally)

**NeurIPS 2021**

[[project page](https://nv-tlabs.github.io/editGAN/)] [[paper](https://arxiv.org/abs/2111.03186)] [[supplementary material](https://nv-tlabs.github.io/editGAN/editGAN_supp_compressed.pdf)]
</div>

### Demos and results

<img src = "https://nv-tlabs.github.io/editGAN/resources/demo2.gif" width="35%"/><img src = "https://nv-tlabs.github.io/editGAN/resources/demo.gif" width="35%"/>

*Left:* The video showcases EditGAN in an interacitve demo tool. *Right:* The video demonstrates EditGAN where we apply multiple edits and exploit pre-defined editing vectors. <u>Note that the demo is accelerated. See paper for run times.</u>

<img src = "https://nv-tlabs.github.io/editGAN/resources/demo_interp.gif" width="35%"/><img src = "https://nv-tlabs.github.io/editGAN/resources/demo_cross.gif" width="28%"/>

*Left:* The video shows interpolations and combinations of multiple editing vectors. *Right:* The video presents the results of applying EditGAN editing vectors on out-of-domain images.

### Requirements

- Python 3.8 is supported.

- Pytorch >= 1.4.0.

- The code is tested with CUDA 10.1 toolkit with Pytorch==1.4.0 and CUDA 11.4  with Pytorch==1.10.0.

- All results in our paper are based on NVIDIA Tesla V100 GPUs with 32GB memory. 

- Set up python environment:
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
- Add the project to PYTHONPATH:
```
export PYTHONPATH=$PWD
```



### Use pre-trained model & Run tool locally 

In the repo, we release our demo tool and pre-trained models for the *car* class. Follow these steps to set up our interactive WebAPP:   

- Download all checkpoints from [checkpoints](https://drive.google.com/drive/folders/1neucNSPp23CeoZs7n5JxrlaCi_rLhwAj?usp=sharing) and put them into a **./checkpoint** folder:

  - **./checkpoint/stylegan_pretrain**: Download the pre-trained checkpoint from [StyleGAN2](https://github.com/NVlabs/stylegan2) and convert the tensorflow checkpoint to pytorch. We also released the converted checkpoint for your convenience. 
  - **./checkpoint/encoder_pretrain**: Pre-trained encoder.
  - **./checkpoint/encoder_pretrain/testing_embedding**: Test image embeddings.
  - **./checkpoint/encoder_pretrain/training_embedding**: Training image embeddings.
  - **./checkpoint/datasetgan_pretrain**: Pre-trained DatasetGAN (segmentation branch).

- Run the app using `python run_app.py`.

- The app is then deployed on the web browser at `locolhost:8888`.



### Training your own model

Here, we provide step-by-step instructions to create a new EditGAN model. We use our fully released *car* class as an example.

- **Step 0:** Train StyleGAN.

  - Download StyleGAN training images from [LSUN](https://www.yf.io/p/lsun).

  - Train your own StyleGAN model using the official [StyleGAN2](https://github.com/NVlabs/stylegan2) code and convert the tensorflow checkpoint to pytorch. Note the specific "stylegan_checkpoint" fields in
    `experiments/datasetgan_car.json ; experiments/encoder_car.json ; experiments/tool_car.json`.
  
  

- **Step 1:** Train StyleGAN Encoder. 

  - Specify location of StyleGAN checkpoint in the "stylegan_checkpoint" field in `experiments/encoder_car.json`.

  - Specify path with training images downloaded in **Step 0** in the "training_data_path" field in `experiments/encoder_car.json`.

  - Run `python train_encoder.py --exp experiments/encoder_car.json`.

    

- **Step 2:** Train DatasetGAN.

  - Specify "stylegan_checkpoint" field in `experiments/datasetgan_car.json`.

  - Download DatasetGAN training images and annotations from [drive](https://drive.google.com/drive/u/1/folders/17vn2vQOF1PQETb1ZgQZV6PlYCkSzSRSa) and fill in "annotation_mask_path" in `experiments/datasetgan_car.json`.

  - Embed DatasetGAN training images in latent space using

    ```
    python train_encoder.py --exp experiments/encoder_car.json --resume *encoder checkppoint* --testing_path data/annotation_car_32_clean --latent_sv_folder model_encoder/car_batch_8_loss_sampling_train_stylegan2/training_embedding --test True
    ```

    and complete "optimized_latent_path" in `experiments/datasetgan_car.json`.

  - Train DatasetGAN (interpreter branch for segmentation) via

    ```
    python train_interpreter.py --exp experiments/datasetgan_car.json
    ```

- **Step 3:** Run the app.

  - Download DatasetGAN test images and annotations from [drive](https://drive.google.com/drive/u/1/folders/1DxHzs5XNn1gLJ_6vAVctdl__nNZerxue). 

  - Embed DatasetGAN test images in latent space via

    ```
    python train_encoder.py --exp experiments/encoder_car.json --resume *encoder checkppoint* --testing_path *testing image path* --latent_sv_folder model_encoder/car_batch_8_loss_sampling_train_stylegan2/training_embedding --test True
    ```

  - Specify the "stylegan_checkpoint", "encoder_checkpoint", "classfier_checkpoint", "datasetgan_testimage_embedding_path" fields in `experiments/tool_car.json`.

  - Run the app via `python run_app.py`.
  
  

### Citations

Please use the following citation if you use our data or code:

```
@inproceedings{ling2021editgan,
  title = {EditGAN: High-Precision Semantic Image Editing}, 
  author = {Huan Ling and Karsten Kreis and Daiqing Li and Seung Wook Kim and Antonio Torralba and Sanja Fidler},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```



### License

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Please see our main [LICENSE](./LICENSE) file.

##### License Dependencies

For any code dependencies related to StyleGAN2, the license is the  Nvidia Source Code License-NC by NVIDIA Corporation, see [StyleGAN2 LICENSE](https://nvlabs.github.io/stylegan2/license.html).

For any code dependencies related to DatasetGAN, the license is the MIT License, see [DatasetGAN LICENSE](https://github.com/nv-tlabs/datasetGAN_release/blob/master/LICENSE.txt).

The dataset of DatasetGAN is released under the [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license by NVIDIA Corporation.

For any code dependencies related to the frontend tool (including html, css and Javascript), the license is the Nvidia Source Code License-NC. To view a copy of this license, visit [./static/LICENSE.md](./static/LICENSE.md). To view a copy of terms of usage, visit [./static/term.txt](./static/term.txt).

