# Adaptive Deep Learning for Breast Cancer Subtype Classification via Multiclass Misprediction Risk Analysis

This project contains the code for our work titled **"Adaptive Deep Learning for Breast Cancer Subtype Classification via Multiclass Misprediction Risk Analysis"** This code enables the detection of mispredictions in any multiclass task trained using any model. In our work, we focus on breast cancer subtype classification. Our general experiments use Densenet121 and EfficientNetb4 as the baseline models, which can be replaced with any deep neural network (DNN) model, including Transformers and Graph Neural Networks.


## Overall Framework
The overall framework of our work is shown below:

![Adaptive Deep Learning for Breast Cancer Subtype Classification via Multiclass Misprediction Risk Analysis
](Risk%20Model.png)




## Data Usage  

- **BRACS:  BReAst Carcinoma Subtyping**  
  [BRACS Dataset](https://www.bracs.icar.cnr.it/)  

- **ICIAR 2018 Grand Challeng on Breast Cancer Histology Dataset**  
  [BACH](https://iciar2018-challenge.grand-challenge.org/Dataset/)  



## Installation
Install the required packages listed in `Requirements.txt`.

## Usage
```bash
PrePreTraining
PrepareRiskData
OneSidedRules
Common
python Main.py 123
