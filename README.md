# AFFGA_Net_plgp

## 1 Dataset

https://sites.google.com/view/plgp-dataset

Put the test set and training set into folders named `train` and `test` respectively, as follows:

```
- F:/sim_grasp3
  - dataset3
    - train
      - 05_0000_0_0_depth.mat
      - 05_0000_0_0_grasp.txt
      - 05_0000_0_0_rgb.png
    - test
      - 05_4535_0_0_depth.mat
      - 05_4535_0_0_grasp.txt
      - 05_4535_0_0_rgb.png
```

## 2 Train Network

> python train_net.py

## 3 Pretrained weights

https://drive.google.com/file/d/19FaSnMmUCa8yjpkX-jgucA0D5wRsNWe0/view?usp=drive_link

## 4 Test Network

Please refer to the testing benchmark of ODG-Generation (https://github.com/dexin-wang/ODG-Generation).
