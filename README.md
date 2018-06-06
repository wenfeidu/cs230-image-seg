# cs230-image-seg

Steps:

1) Download and unzip kaggle images into separate raw_data folder with subfolders for train_color, train_label, and test

2) Run bash files mappings_train and mappings_test to separate train_color, train_label, and test into further subfolders corresponding to video (35 videos for train and 12 videos for test)

3) Run build_dataset.py to separate into train and dev and resize images and labels if necessary

4) Build model code