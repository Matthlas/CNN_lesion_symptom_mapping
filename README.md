# Using CNNs for lesion symptom mapping
Project for the 2018 [HACKUIOWA](https://bigdata.uiowa.edu/) Hackathon.  
  
## Data
We have data for ~400 stroke patients containing manually created lesion masks from MRI images, behaviour scores and LESYMAP results.  
Additionally we have the same data + T1 weighted MRIs for ~100 stroke patients.  

## Goal
The goal is to do lesion symptom mapping using deep learning. We have multiple approaches to do that.

## Approaches

1. Take a vecorized approached and build a simple but deep artificial neural network taking flattened lesion maps as input and one or multiple behavioural scores as label

2. Use Convolutional Neural Networks using the 3D lesion map as input and one or multiple behavioural scores as label.\

3. Use [NiftyNet](http://www.niftynet.io/) which is an open source convolutional neural networks platform for medical image analysis and image-guided therapy to predict behaviour scores. In this case the input could be the entire T1 weighted MRI scan and the lesion map as an additional stream.
