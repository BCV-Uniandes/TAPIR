# Towards Holistic Surgical Scene Understanding

[Natalia Valderrama](https://nfvalderrama.github.io)<sup>1,2</sup>, [Paola Ruiz Puentes](https://paolaruizp.github.io)<sup>1,2*</sup>, Isabela Hernández<sup>1,2*</sup>, [Nicolás Ayobi](https://nayobi.github.io/)<sup>1,2</sup>, Mathilde Verlyck<sup>1,2</sup>, Jessica Santander<sup>3</sup>, Juan Caicedo<sup>3</sup>, Nicolás Fernández<sup>4,5</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1,2</sup> <br/>
<br/>
<sup>*</sup> Equal contribution.<br/>
<sup>1</sup> Center  for  Research  and  Formation  in  Artificial  Intelligence ([CinfonIA](https://cinfonia.uniandes.edu.co/)). <br/> <sup>2</sup> Universidad  de  los  Andes, Bogota, Colombia. <br/>
<sup>3</sup> Fundación Santafé de Bogotá, Bogotá, Colombia<br/>
<sup>4</sup> Seattle Children’s Hospital, Seattle, USA <br/>
<sup>5</sup> University of Washington, Seattle, USA <br/>

- **Oral presentation and best paper nominee** at **Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022**. Proceedings available at [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_42)
- **Preprint** available at [arXiv](https://arxiv.org/abs/2212.04582).

Visit the project in our [website](https://biomedicalcomputervision.uniandes.edu.co/publications/towards-holistic-surgical-scene-understanding/) and our [youtube](https://youtu.be/G4ctkKgRkaY) channel.

## Abstract
<div align="center">
  <img src="images/dataset.jpg"/>
</div><br/>

We present a new experimental framework towards holistic surgical scene understanding. First, we introduce the Phase, Step, Instrument, and Atomic Visual Action Recognition (PSI-AVA) Dataset. PSI-AVA includes annotations for both long-term (Phase and Step recognition) and short-term reasoning (Instrument detection and novel Atomic Action recognition) in robot-assisted radical prostatectomy videos. Second, we present Transformers for Action, Phase, Instrument, and steps Recognition (TAPIR) as a strong baseline for surgical scene understanding. TAPIR leverages our dataset’s multi-level annotations as it benefits from the learned representation on the instrument detection task to improve its classification capacity. Our experimental results in both PSI-AVA and other publicly available databases demonstrate the adequacy of our framework to spur future research on holistic surgical scene understanding.

This repository provides instructions to download the PSI-AVA dataset and run the PyTorch implementation of TAPIR, both presented in the paper Towards Holistic Surgical Scene Understanding, oral presentation at [MICCAI,2022](https://conferences.miccai.org/2022/en/). 

## GraSP dataset and TAPIS

Check out [**GraSP**](https://github.com/BCV-Uniandes/GraSP), an **extended version of our PSI-AVA dataset** that provides **surgical instrument segmentation** annotations and more data. Also check [**TAPIS**](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS), the improved version of our method. GraSP and TAPIS have been published in this [arXiv](https://arxiv.org/abs/2401.11174).

## PSI-AVA

In this [link](http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz), you will find the sampled frames of the original Radical Prostatectomy surgical videos and the annotations that compose the Phases, Steps, Instruments, and Atomic Visual Actions recognition dataset. You will also find the preprocessed data we used for training TAPIR, the instrument detector predictions, and the trained model weights on each task.

We recommend downloading the compressed data archive and extract all files with the following commands:

```sh
$ wget http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz
$ tar -xzvf PSI-AVA.tar.gz
```

After decompressing and extracting all files, the link's data is organized as follows:

```tree
PSI-AVA:
|
|_TAPIR_trained_models
|      |_ACTIONS
|      |    |_Fold1
|      |    |   |_checkpoint_best_actions.pyth
|      |    |_Fold2
|      |        |_checkpoint_best_actions.pyth
|      |_INSTRUMENTS
|      |    ...
|      |_PHASES
|      |    ...
|      |_STEPS
|           ...
|
|_def_DETR_box_ftrs
|     |_fold1
|     |   |_train
|     |   |   |_box_features.pth
|     |   |_val
|     |       |_box_features.pth
|     |_fold2
|         ...
|
|_keyframes
        |_CASE001
        |     |_00000.jpg
        |     |_00001.jpg
        |     |_00002.jpg
        |     ...
        |_CASE002
        |     ...
          ...
```

You will find PSIAVA's data partition and annotations in the [outputs/data_annotations.](https://github.com/BCV-Uniandes/TAPIR/tree/main/outputs/data_annotations) directory.

For **further details on frame preprocessing**, please read the Supplementary Material of our extended article in [arXiv](https://arxiv.org/abs/2401.11174). 
Similarly, if you require **frames extracted at larger frame rates**, the **original surgery videos**, or the **raw frames**, please refer to the [GraSP Repo](https://github.com/BCV-Uniandes/GraSP).

## TAPIR

<div align="center">
  <img src="images/TAPIR.jpg"/>
</div><br/>

### Installation
Please follow these steps to run TAPIR:

```sh
$ conda create --name tapir python=3.8 -y
$ conda activate tapir
$ conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

$ conda install av -c conda-forge
$ pip install -U iopath
$ pip install -U opencv-python
$ pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
$ pip install 'git+https://github.com/facebookresearch/fvcore'
$ pip install 'git+https://github.com/facebookresearch/fairscale'
$ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

$ git clone https://github.com/BCV-Uniandes/TAPIR
$ cd TAPIR
$ pip install -r requirements.txt
```

Our code builds upon [Multi Scale Vision Transformers](https://github.com/facebookresearch/SlowFast)[1]. For more information, please refer to this work.

### Preparing data

First, locate the "keyframes" folder extracted from [PSI-AVA data link](http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz) in the repository's folder ./outputs/PSIAVA/

 ```PSI-AVA/keyframes/* ===> ./outputs/PSIAVA/keyframes/```

Then, locate the instrument features computed by deformable DETR from the "Def_DETR_Box_ftrs" folder extracted from the [PSI-AVA data link](http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz) as follows:
 
 ```PSI-AVA/def_DETR_box_ftrs/fold1/* ===> ./outputs/data_annotations/psi-ava/fold1/*```

 ```PSI-AVA/def_DETR_box_ftrs/fold2/* ===> ./outputs/data_annotations/psi-ava/fold2/*```
 
 Ultimately, the ```outputs``` directory must have the following structure.
 
  ```tree
  outputs
  |_data_annotations
  |      |_psi-ava
  |      |     |_fold1
  |      |     |    |_annotationas
  |      |     |    |    ...
  |      |     |    |_coco_anns
  |      |     |    |    ...
  |      |     |    |_frame_lists
  |      |     |    |    ...
  |      |     |    |_train
  |      |     |    |    |_box_features.pth
  |      |     |    |_val
  |      |     |         |_box_features.pth
  |      |     |_fold2
  |      |          ...
  |      |_psi-ava_extended
  |            ...
  |_PSIAVA
         |_keyframes 
                 |_CASE001
                 |      |_00000.jpg
                 |      |_00001.jpg
                 |      ...
                 |_CASE002
                        ...
                 ...
  ```

### Running the code

First, add this repository in the $PYTHONPATH

```sh
$ export PYTHONPATH=/path/to/TAPIR/slowfast:$PYTHONPATH
```

For training TAPIR run:

```sh
# the Instrument detection or Atomic Action recognition task
$ bash run_examples/mvit_short_term.sh

# the Phases or Steps recognition task
$ bash run_examples/mvit_long_term.sh
```

### Evaluating models

| Task | mAP | config | run file |
| ----- | ----- | ----- | ----- |
| Phases | 56.55 $\pm$ 2.31 | [PHASES](./tree/main/configs/MVIT_PHASES.yaml) | [long_term](./tree/main/run_examples/mvit_long_term.sh) |
| Steps | 45.56 $\pm$ 0.004 | [STEPS](./tree/main/configs/MVIT_STEPS.yaml) | [long_term](./tree/main/run_examples/mvit_long_term.sh) |
| Instruments | 80.85 $\pm$ 1.54 | [TOOLS](./tree/main/configs/MVIT_TOOLS.yaml) | [short_term](./tree/main/run_examples/mvit_short_term.sh) |
| Actions | 28.68 $\pm$ 1.33 | [ACTIONS](./tree/main/configs/MVIT_ACTIONS.yaml) | [short_term](./tree/main/run_examples/mvit_short_term.sh) |

Our pretrained models are stored in [PSI-AVA data link](http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz).

Add this path in the run_examples/mvit_*.sh file corresponding to the task you want to evaluate. Enable the test by setting it in the config **TEST.ENABLE True**

## Contact

If you have any doubts, questions, issues, corrections, or comments, please email n.ayobi@uniandes.edu.co.

## Citing TAPIR

If you use PSI-AVA or TAPIR (or their extended versions, GraSP and TAPIS) in your research, please include the following BibTex citations in your papers.

```BibTeX
@InProceedings{valderrama2020tapir,
      author={Natalia Valderrama and Paola Ruiz and Isabela Hern{\'a}ndez and Nicol{\'a}s Ayobi and Mathilde Verlyck and Jessica Santander and Juan Caicedo and Nicol{\'a}s Fern{\'a}ndez and Pablo Arbel{\'a}ez},
      title={Towards Holistic Surgical Scene Understanding},
      booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
      year={2022},
      publisher={Springer Nature Switzerland},
      address={Cham},
      pages={442--452},
      isbn={978-3-031-16449-1}
}

@article{ayobi2024pixelwise,
      title={Pixel-Wise Recognition for Holistic Surgical Scene Understanding}, 
      author={Nicolás Ayobi and Santiago Rodríguez and Alejandra Pérez and Isabela Hernández and Nicolás Aparicio and Eugénie Dessevres and Sebastián Peña and Jessica Santander and Juan Ignacio Caicedo and Nicolás Fernández and Pablo Arbeláez},
      year={2024},
      url={https://arxiv.org/abs/2401.11174},
      eprint={2401.11174},
      journal={arXiv},
      primaryClass={cs.CV}
}
```

## References

[1] H. Fan, Y. Li, B. Xiong, W.-Y. Lo, C. Feichtenhofer, ‘PySlowFast’, 2020. https://github.com/facebookresearch/slowfast.