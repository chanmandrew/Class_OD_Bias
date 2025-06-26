## Quantifying Sex Bias in Deep Learning Neural Networks for Pneumonia Detection on Chest X-Rays

Although artificial intelligence (AI) systems using deep learning (DL) have demonstrated great promise for automating expert-level diagnosis of disease in medical imaging, they have also demonstrated the potential for bias against underrepresented or historically disadvantaged groups. For example, a recent study has shown that DL classification models trained to identify diseases on the basis of chest radiographs/x-rays (CXR) using images from exclusively males or exclusively females systematically had worse diagnostic accuracy on the minority sex. Biased DL models are concerning because they have the potential to perpetuate pre-existing healthcare inequities, such as unequal access to care (which could present as underrepresentation in datasets used to develop DL models). Although prior work has shown that DL classification models can be biased when trained with unbalanced sex representation, it is unclear if other types of DL diagnostic models are as susceptible to these biases, particularly those that use richer disease annotations at the pixel label, such as object detection or segmentation models. In this project, we quantify the sex-based biases of DL object detection models for pneumonia diagnosis on CXRs and compare the results to that of classification models. We will perform evaluations of DL models on ‘real-world’ datasets as provided by well-known data science competitions. We hypothesize that DL object detection models will show fewer degrees of sex-based biases compared to DL classification models. Our experiments are underway and we will present them at the forum. Ultimately, this study will improve our understanding of the potential pitfalls of using DL for medical image diagnosis, as well as the relative advantages of using different DL models with regard to bias

For those interested, this project was inspired by https://www.pnas.org/doi/10.1073/pnas.1919012117

### Specific Aims
The overarching goal of this project is to evaluate whether training DL models with pixel-level annotations (i.e., pneumonia bounding boxes) and unbalanced sex representation will result in models with performance biases against the underrepresented sex. Our specific aims are as follows:

1. To compare the degree of sex-based bias of pneumonia object detection DL models to equivalent DL classification models when trained on a ‘real-world’ Kaggle data science competition dataset.
2. To quantify the impact of sex imbalance in CXR datasets on the degree of sex-based biased in DL pneumonia object detection and classification models.

For the second aim, we will be training datasets on various different gender ratios (0%, 25%, 50%, 75%, and 100% female) and seeing how the two models perform on each gender. 

Data: https://1drv.ms/u/s!AiBrH5HLacq2gZ5XC_9qfUnZDf1zww?e=lNfleQ
