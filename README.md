# 1. Introduction: 
This is the official code for the paper ‘Fine Back Surfaces Oriented Human Reconstruction 
for Single RGB-D Images’, which can obtain robust human surface reconstruction 
with a single RGB-D image. The article is published by Pacific Graphics 2023.

 
# 2. Requirements: 

Python 
Pytorch 
Trimesh 
Python-opencv 
Numpy 
 

# 3. Pretrained models 

First, please download the pretrained models from this link: 
https://drive.google.com/file/d/1asdOmDqfNDSTNbJdCx5wE1wdJhBEyFX2/vie
w?usp=drive_link 

Then, put the pretrained models in {myproject}/model/ 

# 4. Test 

Generate the csv file for the demo images in {myproject}/data. 

 
Adjust the number of file names in the createTestcsv file based on the format and 
number of file names in the example shown in demo to create a csv file that you want 
to read. 
 
## 
cd data 
python createTestCsv.py 
cd .. 
## 
python test.py 
## 
Results are shown in {myproject}/results/obj. 

 

# 5. Citation 

Please cite our paper if you feel this code is useful. Thanks. 
Xianyong Fang, Yu Qian, Jinshen He, Linbo Wang, Zhengyi Liu. Fine Back Surfaces 
Oriented Human Reconstruction for Single RGB-D Images, Pacific Graphics 2023.

@article {FangQHWL2023fineback,

journal = {Computer Graphics Forum},

title = {{Fine Back Surfaces Oriented Human Reconstruction for Single RGB-D Images}},

author = {Fang, Xianyong and Qian, Yu and He, Jinshen and Wang, Linbo and Liu, Zhengyi},

year = {2023},

publisher = {The Eurographics Association and John Wiley & Sons Ltd.},

ISSN = {1467-8659},

DOI = {10.1111/cgf.14971}

}
