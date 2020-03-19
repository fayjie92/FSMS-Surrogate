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

#### Please consider starring us, if you found it useful. Thanks

## Updates
- March 7, 2020: Implementation code is available now.
</br>

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras version 2.2.0
- tensorflow backend version 1.13.1


## Run Demo
The implementation code is availabel in Source Code folder.</br>
1- Download the FSS1000 dataset from [this](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI) link and extract the dataset.</br>
2- Run `Train_DOGLSTM.py` for training Scale Space Encoder model using k-shot episodic training. The model will be train for 50 epochs and for each epoch it will itterate 1000 episodes to train the model. The model will saves validation performance history and the best weights for the valiation set. It also will report the MIOU performance on the test set. The model by default will use VGG backbone with combining Block 3,4 and 5 but other combination can be call in creatign the model. It is also possible to use any backbone like Resnet, Inception and etc.... </br>
3- Run `Train_weak.py` for training Scale Space Encoder model using k-shot episodic training and evaluatign on the weak annotation test set. This code will use weaklly annotated bouning box as a label for the support set on test time.

Notice: `parser_utils.py` can be used for hyper parameter setting and defining data set address, k-shot and n-way.

## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/fewshot-segmentation/blob/master/githubimages/Figure1.png)

### Structure of the Proposed Scale Space encoder for reducing texture bias effect
![Diagram of the SSR](https://github.com/rezazad68/fewshot-segmentation/blob/master/githubimages/Figure2.png)

### Visual representation of 21 classes from 1000-class dataset with their masks and generated bounding box [Download link](https://github.com/rezazad68/fewshot-segmentation/raw/master/FSS-1000%20Bounding%20Box%20Annotation.zip)
![Bounding Box annotation for FSS-1000](https://github.com/rezazad68/fewshot-segmentation/blob/master/githubimages/Weak%20Annotation%20samples%20for%20FSS1000.jpg)
