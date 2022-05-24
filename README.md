# MSO
Pytorch Code for MSO: Multi-Feature Space Joint Optimization Network for RGB-Infrared Person Re-Identification[1]. 

We adopt the AGW[2] as the backbone. 

|Datasets    | Eage_method| Rank@1  | mAP |  mINP | 
| --------   | -----    | -----  |  -----  | ----- |
|#SYSU-MM01  | Laplace | ~ 49.12%  | ~ 48.91% | ~35.50% | 
|#SYSU-MM01  | Sobel   | ~ 56.30%  | ~ 54.55% | ~40.99% | 
|#SYSU-MM01  | Prewitt | ~ 54.36%  | ~ 52.61% | ~38.30% | 

* The results may have some fluctuation due to random spliting.

### 1. Prepare the datasets.
  
-  SYSU-MM01 Dataset [3]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training.
  Train a model by
  ```bash
python train.py --dataset sysu --lr 0.1 --method agw --gpu 0
```

  - `--dataset`: which dataset "sysu".

  - `--lr`: initial learning rate.
  
  -  `--method`: method to run or baseline.
  
  - `--gpu`:  which gpu to run.

Select eage extracting method in PEF.py.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

**Sampling Strategy**: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image. Details can be found in Line 302-307 in `train.py`.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 dataset by 
  ```bash
python test.py --mode all --resume 'model_path' --gpu 0 --dataset sysu
```
  - `--dataset`: which dataset "sysu".
  
  - `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

  - `--resume`: the saved model path.
  
  - `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```

###  5. References.
[1] Gao, Yajun, et al. "MSO: Multi-Feature Space Joint Optimization Network for RGB-Infrared Person Re-Identification." Proceedings of the 29th ACM International Conference on Multimedia(MM), 2021.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible
light and thermal cameras. Sensors, 17(3):605, 2017.

[3] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
