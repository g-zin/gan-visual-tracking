# **Generative Adversarial Networks for Online Visual Object Tracking Systems**

## **Introduction:**
This project is to build visual object tracking benchmark using generative adversarial networks to track any moving target in a sequence without prior traning on the target.

This implemntation is based on [MDNET](https://github.com/HyeonseobNam/py-MDNet), [VITAL](https://github.com/abnerwang/py-Vital) and [RT-MDNet](https://github.com/sydney0zq/RT-MDNet-OPN). 

I modified the architecture of the baseline to get MDGanet, ROIAL-MDNet, MDResNet and MDResGaNet trackers.

For more information:

- [Thesis](https://scholars.wlu.ca/etd/2196/)

- [Presentation](https://github.com/g-zin/gan-visual-tracking/files/4535249/GANs.for.online.viusal.object.tracking.systems_presentation.pdf)

- [Demo](https://youtu.be/s2DPnLAxFuQ)



## **Usage**
### ***For Tracking***

```
python run_trackers.py  -t <trackers> -s <sequences> -e <evaltypes> -n <testname> 
```
For example to run ROIAL tracker: 
```
python run_trackers.py  -t RoialMDNet -s Basketball -e OPE -n tb50 
```


### ***For Pre-Training***

- Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"

- Pretraining on VOT-OTB:
  - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/VOT/vot201x"
- Pretraining on ImageNet-VID
  - Download [ImageNet-VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid) dataset into "datasets/ILSVRC"



