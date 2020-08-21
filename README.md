# [Semi-supervised Few-Shot Learning for Medical Image Segmentation]()

Implementation code for Semi-supervised approach for few-shot semantic medical image segmentation. This method is the first attempt to apply the episodic training paradigm for few-shot medical image segmentation. This method also enriches the feature representation in an unsupervised manner. More specifically, this method boosts the encoder representation with including surrogate task. Experimental results on two well-know Skin Lesion Segmentation data sets have been demonstrated that the proposed method produces promising results.<!--  If this code helps with your research please consider citing the following paper:
</br>
<!-- 
>[Abdur. R. Fayjie](https://sites.google.com/site/abdurrfayjie/),
[Marco Pedersoli](https://scholar.google.com/citations?user=aVfyPAoAAAAJ&hl=en),
[Claude Kauffman](https://radiologie.umontreal.ca/departement/les-professeurs/profil/kauffmann-claude/in15322/),
[Ismail Ben Ayed](https://scholar.google.com/citations?hl=en&user=29vyUccAAAAJ&view_op=list_works&sortby=pubdate) and
[Jose Dolz](https://scholar.google.ca/citations?user=yHQIFFMAAAAJ&hl=en) 
"Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020, download [link](https://arxiv.org/pdf/2003.08462.pdf).

## Updates
- March 20, 2020: Implementation code is available now.
-->
</br>

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras version 2.2.0
- tensorflow backend version 1.13.1


## Run Demo
Please follow the bellow steps to run the code.</br>
1- Download the FSS1000 dataset from [this](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI) link and extract the dataset to a folder name `fss_dataset`.</br>
2- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders to a folder `ISIC2018`. </br>
3- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract to a folder `dataset_PH2Datasetisic18`. </br>
4- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train(unlabeled) and test sets. </br>
5- Run `Prepare_ph2.py` for data preperation and providing test sets. </br>
6- Run `Train_evaluate.py` for training and evaluation. 

## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/FSMS-Surrogate-/blob/master/githubimages/Method.png)
### Quantitive result 
![Segmentation Result](https://github.com/rezazad68/FSMS-Surrogate-/blob/master/githubimages/Table.jpg)

### Visual representation of the segmentation results on both ph2 and ISIC dataset
![Segmentation Result](https://github.com/rezazad68/FSMS-Surrogate-/blob/master/githubimages/Result.png)

