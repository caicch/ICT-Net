# ICT-Net
Interactive Change-aware Transformer Network for Remote Sensing Image Change Captioning

# Set up
Follow the baseline method [[Link Text](https://github.com/Chen-Yang-Liu/RSICC)] to download the dataset and set up conda env. 

# Training
Use python train.py to run the code //
python train.py  --data_folder ./data/ --savepath ./models_checkpoint/

# Evaluate //

python eval.py --data_folder ./data/ --path ./models_checkpoint/ --Split TEST

# Citation:
@Article{rs15235611,
AUTHOR = {Cai, Chen and Wang, Yi and Yap, Kim-Hui},
TITLE = {Interactive Change-Aware Transformer Network for Remote Sensing Image Change Captioning},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {23},
ARTICLE-NUMBER = {5611},
URL = {https://www.mdpi.com/2072-4292/15/23/5611},
ISSN = {2072-4292},
DOI = {10.3390/rs15235611}
}

# Reference: 
https://github.com/Chen-Yang-Liu/RSICC

