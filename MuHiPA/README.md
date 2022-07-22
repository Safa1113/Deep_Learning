# MuHiPA: Multimodal Hierarchical Reasoning based on Perception and Action Cycle for Visual Question Answering

In vision-language tasks, most recent frameworks exploit large scale datasets and share transformer-like reasoning backbones which contribute less to multi-modality fusion components. For valid progress in the vision-language community, we suggest the necessity of examining diverse methodologies and evaluation metrics of joint representations especially for understanding tasks. In this view, we introduce Multimodal Hierarchical Reasoning based on Perception Action Cycle (MuHiPA) framework for tackling visual question answering (VQA) tasks. MuHiPA takes one step closer to human-like thought process. It integrates the perception action cycle principle which explains the learning mechanism of humans about the surrounding world. Therefore, MuHiPA comprehends the visual modality through three stages: selection, organization, and interpretation. Moreover, vision and language modalities are interpreted dependently in a cyclic and hierarchical way through the entire framework. For joint representations assessment, we argue that question-image pairs of the same answer ought to have similar representations eventually. Hence, we test them against the standard deviation of cosine similarity and of Manhattan distance. We demonstrate that our framework scores lower standard deviations compared to other VQA frameworks indicating adequate reasoning. For further assessment of our approach for visual modality comprehension, the organization stage is incorporated into a visual relationship detection framework. The proposed incorporation of perception action cycle principle achieves state-of-the-art results on TDIUC and VRD datasets while it achieves competitive results on the VQA 2.0 dataset. 

The model is trained on 63 GB GPU. One epoch takes from 15 to 20 mins. All models are trained to 25 epochs. Accuracy metrics codes are provided by datasets creators.

#### Summary

* [Installation](#installation)
    * [Python 3 & Anaconda](#1-python-3--anaconda)
    * [Download datasets](#3-download-datasets)
* [Quick start](#quick-start)
    * [Train a model](#train-a-model)
    * [Evaluate a model](#evaluate-a-model)
* [Reproduce results](#reproduce-results)
    * [VQA2](#vqa2-dataset)
    * [TDIUC](#tdiuc-dataset)
    * [VRD](#vrd-dataset)



## Installation

### 1. Python 3 & Anaconda

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

### 2. As standalone project

```
conda create --name muhipa python=3.7
source activate muhipa
cd MuHiPA_
pip install -r requirements.txt
```

### 3. Download datasets

Download annotations, images and features for VQA experiments:
```
bash MuHiPA/datasets/scripts/download_vqa2.sh
bash MuHiPA/datasets/scripts/download_vgenome.sh
bash MuHiPA/datasets/scripts/download_tdiuc.sh
bash MuHiPA/datasets/scripts/download_vrd.sh
```

**Note:** The features have been extracted from a pretrained Faster-RCNN with caffe. We don't provide the code for pretraining or extracting features for now.


## Quick start

### Train a model

You can train our best model on VQA2 by running:
```
python -m run -o MuHiPA/options/vqa2/MuHiPA.yaml
```
Then, several files are going to be created in `logs/vqa2/EMuRelPAtest1`:
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_engine.pth.tar (checkpoints of best epoch)
- ckpt_best_eval_epoch.accuracy_top1_model.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_optimizer.pth.tar


### Evaluate a model

At the end of the training procedure, you can evaluate your model on the testing set. 
```
python -m run -o logs/vqa2/MuHiPAAtest1/options.yaml --exp.resume best_accuracy_top1 --dataset.train_split --dataset.eval_split test --misc.logs_name test
```

## Reproduce results

### VQA2 dataset

VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
-265,016 images (COCO and abstract scenes)
-At least 3 questions (5.4 questions on average) per image
-10 ground truth answers per question
-3 plausible (but likely incorrect) answers per question
-Automatic evaluation metric

VQA 2.0 has three splits. The training split contains 82,783 images with 443,757 questions. The validation split has 40,504 images with 214,354 questions. The test-standard split contains 81,434 images with 447,793 questions.

#### Training and evaluation (train/val)

We use this simple setup to tune our hyperparameters on the valset for MuHiPA.

```
python -m run -o MuHiPA/options/vqa2/MuHiPA.yaml --exp.dir logs/vqa2/MuHiPAtest
```

For the baseline model MuRel, we use 

```
python -m run -o MuHiPA/options/vqa2/murel.yaml
```

### TDIUC dataset


The TDIUC dataset has 167,437 images, 1,654,167 questions, 12 question types and 1,618 unique answers.
Question Type - Number of Questions - Number of Unique Answers: 
Scene Recognition - 66,706 - 83
Sport Recognition - 31,644 - 12
Color Attributes - 195,564 - 16
Other Attributes - 28,676 - 625
Activity Recognition - 8,530 - 13
Positional Reasoning - 38,326 - 1,300
Sub. Object Recognition - 93,555 - 385
Absurd - 366,654 - 1
Utility/Affordance - 521 - 187
Object Presence - 657,134 - 2
Counting - 164,762 - 16
Sentiment Understanding - 2,095 - 54
Grand Total - 1,654,167 - 1,618


#### Training and evaluation (train/val/test)

The full training set is split into a trainset and a valset. At the end of the training, we evaluate our best checkpoint on the testset. The TDIUC metrics are computed and displayed at the end of each epoch. They are also stored in `logs.json` and `logs_test.json`.


```
python -m run -o MuHiPA/options/tdiuc/MuHiPA.yaml --exp.dir logs/tdiuc/MuHiPAtest

```

For the baseline model MuRel, we use :

```
python -m run -o MuHiPA/options/tdiuc/murel.yaml

```

### VRD dataset


The Visual Relationship Dataset (VRD) contains 4000 images for training and 1000 for testing annotated with visual relationships. Bounding boxes are annotated with a label containing 100 unary predicates. These labels refer to animals, vehicles, clothes and generic objects. Pairs of bounding boxes are annotated with a label containing 70 binary predicates. These labels refer to actions, prepositions, spatial relations, comparatives or preposition phrases. The dataset has 37993 instances of visual relationships and 6672 types of relationships. 1877 instances of relationships occur only in the test set and they are used to evaluate the zero-shot learning scenario.


#### Training and evaluation (train/val/test)

The full training set is split into a trainset and a valset. At the end of the training, we evaluate our best checkpoint on the testset. 


```
python -m run -o MuHiPA/options/vrd/muhipa.yaml --exp.dir logs/vrd/MuHiPAtest

```

For the baseline model Block, we use :

```
python -m run -o MuHiPA/options/vrd/block.yaml

```

## Pretrained models

To evaluate pretrianed models of MuHiPA on Tdiuc:

```
python -m run -o logs/tdiuc/MuHiPA/options.yaml --exp.resume best_eval_epoch.accuracy_top1 --dataset.train_split --dataset.eval_split test --misc.logs_name test

```

To evaluate pretrianed models of MuHiPA on VQA2:

```
python -m run -o logs/vqa2/MuHiPA/options.yaml --exp.resume best_eval_epoch.accuracy_top1 --dataset.train_split --dataset.eval_split test --misc.logs_name test
```



