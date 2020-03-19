# [Semi-supervised Few-Shot Learning for Medical Image Segmentation]()

Implementation code for Semi-supervised approach for few-shot semantic medical image segmentation. This method is the first attempt to applies episodic training process for learning few-shot medical image segmentation. This method also enritches the festure representation using unsupervised manner. More specific, this method boosts the encoder representation with including surrogate task. Experimental results on two well-know Skin Lesion Segmentation data sets have been demonstrated that the proposed method produces promissing results. If this code helps with your research please consider citing the following paper:
</br>
>[Abdur. R. Fayjie](https://sites.google.com/site/abdurrfayjie/),
[R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate),
[Claude Kauffman](https://radiologie.umontreal.ca/departement/les-professeurs/profil/kauffmann-claude/in15322/),
[Ismail Ben Ayed](https://scholar.google.com/citations?hl=en&user=29vyUccAAAAJ&view_op=list_works&sortby=pubdate),
[Marco Pedersoli](https://scholar.google.com/citations?user=aVfyPAoAAAAJ&hl=en) and
[Jose Dolz](https://scholar.google.ca/citations?user=yHQIFFMAAAAJ&hl=en) 
"Semi-supervised Few-Shot Learning for Medical Image Segmentation", arXiv preprint arXiv, 2020, download [link]().

## Updates
- March 20, 2020: Implementation code is available now.
</br>

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras version 2.2.0
- tensorflow backend version 1.13.1


## Run Demo
Please follow the bellow steps to run the code.</br>
1- Download the FSS1000 dataset from [this](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI) link and extract the dataset to a folder name `dataset_FSS1000`.</br>
2- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders to a folder `dataset_isic18`. </br>
3- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train(unlabeled) and test sets. </br>

****4- Download the Ph2 dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders to a folder `dataset_ph2`. </br>
5- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train(unlabeled) and test sets. </br>

2- Run `Train_DOGLSTM.py` for training and evaluation. 

## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/FSMS-Surrogate-/blob/master/githubimages/Main%20model.png)

### Visual representation of the segmentation results on both ph2 and ISIC dataset
![Segmentation Result](https://github.com/rezazad68/FSMS-Surrogate-/blob/master/githubimages/Result.png)
