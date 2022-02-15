# Learning a Prototype Discriminator with RBF for Multimodal Image Synthesis

The PyTorch implements of Learning a Prototype Discriminator with RBF for Multimodal Image Synthesis.

**The overview of our PT-GAN framework.**
<img src="images/framework.jpg"/>


Our method can synthesis clear and nature images and outperforms other state-of-the-art methods on many datasets.

Experiment results on **BraTS2020** dataset.
<img src="images/comparison_brats.jpg"/>

Experiment results on **ISLES2015** dataset.
<img src="images/comparison_isles.jpg"/>

Experiment results on **CMU Multi-PIE** dataset.
<img src="images/comparison_multipie.jpg"/>

## Environment
```
python              3.8.10
pytorch             1.8.1
torchvision         0.9.1
tqdm                4.62.1
numpy               1.20.3
SimpleITK           2.1.0
scikit-learn        0.24.2
opencv-python       4.5.3.56
easydict            1.9
tensorboard         2.5.0
Pillow              8.3.1
```
## Datasets
Download the datasets from the official way and rearrange the files to the following structure.
The dataset path can be modified in the PT-GAN/options/\*.yaml file.
### BraTS2020
```
MICCAI_BraTS2020_TrainingData
├── flair
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_002_flair.nii.gz
│   ├── BraTS20_Training_003_flair.nii.gz
│   ├── ...
├── t2
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_002_t2.nii.gz
│   ├── BraTS20_Training_003_t2.nii.gz
│   ├── ...
├── t1
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_002_t1.nii.gz
│   ├── BraTS20_Training_003_t1.nii.gz
│   ├── ...
├── t1ce
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_002_t1ce.nii.gz
│   ├── BraTS20_Training_003_t1ce.nii.gz
│   ├── ...
```
### ISLES2015
```
SISS2015_Training
├── 1
│   ├── VSD.Brain.XX.O.MR_T2.70616
│        ├── VSD.Brain.XX.O.MR_T2.70616.nii
│   ├── VSD.Brain.XX.O.MR_T1.70615
│        ├── VSD.Brain.XX.O.MR_T1.70615.nii
│   ├── VSD.Brain.XX.O.MR_Flair.70614
│        ├── VSD.Brain.XX.O.MR_Flair.70614.nii
│   ├── VSD.Brain.XX.O.MR_DWI.70613
│        ├── VSD.Brain.XX.O.MR_DWI.70613.nii
├── 2
│   ├── VSD.Brain.XX.O.MR_T2.70622
│        ├── VSD.Brain.XX.O.MR_T2.70622.nii
│   ├── VSD.Brain.XX.O.MR_T1.70621
│        ├── VSD.Brain.XX.O.MR_T1.70621.nii
│   ├── VSD.Brain.XX.O.MR_Flair.70620
│        ├── VSD.Brain.XX.O.MR_Flair.70620.nii
│   ├── VSD.Brain.XX.O.MR_DWI.70619
│        ├── VSD.Brain.XX.O.MR_DWI.70619.nii
├── 3
│   ├── ...
```

### CMU-MultiPIE
```
MultiPIE_Illumination
├── train
│   ├── l45
│        ├── 001.png
│        ├── 002.png
│        ├── ...
│   ├── l90
│        ├── 001.png
│        ├── 002.png
│        ├── ...
│   ├── r45
│        ├── 001.png
│        ├── 002.png
│        ├── ...
│   ├── r90
│        ├── 001.png
│        ├── 002.png
│        ├── ...
│   ├── front
│        ├── 001.png
│        ├── 002.png
│        ├── ...
├── test
│   ├── ...
```

## Checkpoints
Our pre-trained models are available at: [Google Drive](https://drive.google.com/file/d/1CqOQHCK9d_811Vw9gIr3rgjGZX7IB3Ke/view?usp=sharing) | [OneDrive](https://1drv.ms/u/s!AsJlLKv0WJvdlG8bDfUNqiHJ3_SP?e=NcjVSx) | [Baidu Drive](https://pan.baidu.com/s/1VDgTMA-umv8BO2T-guBl-A) Password: 7gcx

## Test
Edit the .yaml file of the corresponding dataset for testing configuration and run the following command to test.
```
python test.py options/brats.yaml
```
The training code will be released after the paper is accepted.
