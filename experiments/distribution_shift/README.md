# Evalutating Distribution Shifts
## Code Base Information: DomainBed

Our code is based on the [DomainBed](https://github.com/facebookresearch/DomainBed), a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). 

## License
This DomainBed is released under the MIT license. 

## Usage
please read main_generalization.py

## Available algorithms

The [currently available algorithms](distribution_shift/algorithms.py) are:

* Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Invariant Risk Minimization (IRM, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Group Distributionally Robust Optimization (GroupDRO, [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731))
* Interdomain Mixup (Mixup, [Yan et al., 2020](https://arxiv.org/abs/2001.00677))
* Marginal Transfer Learning (MTL, [Blanchard et al., 2011-2020](https://arxiv.org/abs/1711.07910))
* Meta Learning Domain Generalization (MLDG, [Li et al., 2017](https://arxiv.org/abs/1710.03463))
* Maximum Mean Discrepancy (MMD, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
* Deep CORAL (CORAL, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719))
* Domain Adversarial Neural Network (DANN, [Ganin et al., 2015](https://arxiv.org/abs/1505.07818))
* Conditional Domain Adversarial Neural Network (CDANN, [Li et al., 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf))
* Style Agnostic Networks (SagNet, [Nam et al., 2020](https://arxiv.org/abs/1910.11645))
* Adaptive Risk Minimization (ARM, [Zhang et al., 2020](https://arxiv.org/abs/2007.02931)), contributed by [@zhangmarvin](https://github.com/zhangmarvin)
* Variance Risk Extrapolation (VREx, [Krueger et al., 2020](https://arxiv.org/abs/2003.00688)), contributed by [@zdhNarsil](https://github.com/zdhNarsil)
* Representation Self-Challenging (RSC, [Huang et al., 2020](https://arxiv.org/abs/2007.02454)), contributed by [@SirRob1997](https://github.com/SirRob1997)
* Spectral Decoupling (SD, [Pezeshki et al., 2020](https://arxiv.org/abs/2011.09468))
* Learning Explanations that are Hard to Vary (AND-Mask, [Parascandolo et al., 2020](https://arxiv.org/abs/2009.00329))
* Out-of-Distribution Generalization with Maximal Invariant Predictor (IGA, [Koyama et al., 2020](https://arxiv.org/abs/2008.01883))
* Gradient Matching for Domain Generalization (Fish, [Shi et al., 2021](https://arxiv.org/pdf/2104.09937.pdf))
* Self-supervised Contrastive Regularization (SelfReg, [Kim et al., 2021](https://arxiv.org/abs/2104.09841))


The implementations use ResNet50 / ResNet18 networks ([He et al., 2015](https://arxiv.org/abs/1512.03385)) 

Example Commands:
```sh
clear && CUDA_VISIBLE_DEVICES=3 python main_generalization.py --num-domains 2 --algorithm ERM 

clear && CUDA_VISIBLE_DEVICES=4 python main_generalization.py --num-domains 2 --algorithm GroupDRO 

clear && CUDA_VISIBLE_DEVICES=5 python main_generalization.py --num-domains 2 --algorithm IRM 

clear && CUDA_VISIBLE_DEVICES=6 python main_generalization.py --num-domains 2 --algorithm CORAL 

clear && CUDA_VISIBLE_DEVICES=7 python main_generalization.py --num-domains 2 --algorithm CDANN 

```