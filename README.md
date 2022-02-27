# MetaShift: A Dataset of Datasets for Evaluating Distribution Shifts and Training Conflicts 

[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://metashift.readthedocs.io/)
[![Analytics](https://ga-beacon.appspot.com/UA-158040863-4/github.com/Weixin-Liang/MetaShift/README.md?pixel)](https://GitHub.com//Weixin-Liang/MetaShift/)
[![Documentation Status](https://readthedocs.org/projects/metashift/badge/?version=latest)](http://metashift.readthedocs.io/?badge=latest)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![OpenReview](https://img.shields.io/badge/OpenReview-MTex8qKavoS-green.svg)](https://openreview.net/forum?id=MTex8qKavoS)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.8-red.svg)](https://shields.io/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
[![GitHub stars](https://img.shields.io/github/stars/Weixin-Liang/MetaShift.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/Weixin-Liang/MetaShift/)


This repo provides the PyTorch source code of our paper: 
[MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts](https://openreview.net/forum?id=MTex8qKavoS) (ICLR 2022). 
[[PDF]](https://openreview.net/forum?id=MTex8qKavoS)

**Project website:** https://MetaShift.readthedocs.io/ 


```
@InProceedings{liang2022metashift,
  title={MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts},
  author={Weixin Liang and James Zou},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=MTex8qKavoS}
}
```



This repo provides the scripts for generating the proposed MetaShift, which offers a resource of 1000s of distribution shifts.   

<!-- and the PyTorch source code for the experiments of evaluating distribution shifts and training conflicts.  -->


## Abstract
*Understanding the performance of machine learning model across diverse data distributions is critically important for reliable applications. Motivated by this, there is a growing focus on curating benchmark datasets that capture distribution shifts. While valuable, the existing benchmarks are limited in that many of them only contain a small number of shifts and they lack systematic annotation about what is different across different shifts. We present MetaShift---a collection of 12,868 sets of natural images across 410 classes---to address this challenge. We leverage the natural heterogeneity of Visual Genome and its annotations to construct MetaShift. The key construction idea is to cluster images using its metadata, which provides context for each image (e.g. cats with cars or cats in bathroom) that represent distinct data distributions. MetaShift has two important benefits: first it contains orders of magnitude more natural data shifts than previously available. Second, it provides explicit explanations of what is unique about each of its data sets and a distance score that measures the amount of distribution shift between any two of its data sets. We demonstrate the utility of MetaShift in benchmarking several recent proposals for training models to be robust to data shifts. We find that the simple empirical risk minimization performs the best when shifts are moderate and no method had a systematic advantage for large shifts. We also show how MetaShift can help to visualize conflicts between data subsets during model training.*

<p align='center'>
  <img width='100%' src='./docs/figures/MetaShift-Examples.jpg'/>
<b>Figure 1: Example Cat vs. Dog Images from MetaShift. </b> For each class, MetaShift provides many subsets of data, each of which corresponds different contexts (the context is stated in parenthesis). 
</p>


<p align='center'>
  <img width='100%' src='./docs/figures/MetaShift-InfoGraphic.jpg'/>
<b>Figure 2: Infographics of MetaShift. </b> 

<p align='center'>
  <img width='100%' src='./docs/figures/Cat-MetaGraph.jpg'/>
<b>Figure 3: Meta-graph: visualizing the diverse data distributions within the “cat” class.  </b> 


## Repo Structure Overview
```plain
.
├── README.md
├── dataset/
    ├── meta_data/ 
    ├── generate_full_MetaShift.py
    ├── ...         
├── experiments/
    ├── subpopulation_shift/              
        ├── main_generalization.py
        ├── ...
```
The `dataset` folder provides the script for generating MetaShift. 
The `experiments` folder provides the expriments on MetaShift in the paper. 



## Dependencies
* Python 3.6.13 (e.g. `conda create -n venv python=3.6.13`)
* PyTorch Version:  1.4.0
* Torchvision Version:  0.5.0

### Download Visual Genome

We leveraged the natural heterogeneity of [Visual Genome](https://visualgenome.org) and its annotations to construct MetaShift. Download the pre-processed and cleaned version of Visual Genome by [GQA](https://arxiv.org/pdf/1902.09506.pdf). 

- Download image files (~20GB) and scene graph annotations: 
```
wget -c https://nlp.stanford.edu/data/gqa/images.zip
unzip images.zip -d allImages
wget -c https://nlp.stanford.edu/data/gqa/sceneGraphs.zip  
unzip sceneGraphs.zip -d sceneGraphs
```

- After this step, the base dataset file structure should look like this:
```
/data/GQA/
    allImages/
        images/
            <ID>.jpg
    sceneGraphs/
        train_sceneGraphs.json
        val_sceneGraphs.json
```

- Specify local path of Visual Genome
Extract the files, and then specify the folder path 
(e.g., `IMAGE_DATA_FOLDER=/data/GQA/allImages/images/`) in [Constants.py](./dataset/Constants.py). 


## Generate the Full MetaShift Dataset (subsets defined by contextual objects)

### Understanding `dataset/meta_data/full-candidate-subsets.pkl`
The metadata file `dataset/meta_data/full-candidate-subsets.pkl` is the most important piece of metadata of MetaShift, which provides the full subset information of MetaShift. To facilitate understanding, we have provided a notebook `dataset/understanding_full-candidate-subsets-pkl.ipynb` to show how to extract information from it. 

Basically, the pickle file stores a `collections.defaultdict(set)` object, which contains *17,938* keys. Each key is a string of the subset name like `dog(frisbee)`, and the corresponding value is a list of the IDs of the images that belong to this subset. The image IDs can be used to retrieve the image files from the Visual Genome dataset that you just downloaded. In our current version, *13,543* out of *17,938* subsets have more than 25 valid images. In addition, `dataset/meta_data/full-candidate-subsets.pkl` is drived from the [scene graph annotation](https://nlp.stanford.edu/data/gqa/sceneGraphs.zip), so check it out if your project need additional information about each image. 



### Generate Full MetaShift

Since the total number of all subsets is very large, all of the following scripts only generate a subset of MetaShift. As specified in [dataset/Constants.py](./dataset/Constants.py), we only generate MetaShift for the following classes (subjects). You can add any additional classes (subjects) into the list. See [dataset/meta_data/class_hierarchy.json](./dataset/meta_data/class_hierarchy.json) for the full object vocabulary and its hierarchy. 
`SELECTED_CLASSES = [
    'cat', 'dog',
    'bus', 'truck',
    'elephant', 'horse',
    'bowl', 'cup',
    ]` 


In addition, to save storage, all copied images are symbolic links. You can set `use_symlink=True` in the code to perform actual file copying. If you really want to generate the **full** MetaShift, then set `ONLY_SELECTED_CLASSES = True` in [dataset/Constants.py](./dataset/Constants.py). 

```sh
cd dataset/
python generate_full_MetaShift.py
```

The following files will be generated by executing the script. Modify the global varaible `SUBPOPULATION_SHIFT_DATASET_FOLDER` to change the destination folder.  

```plain
/data/MetaShift/MetaDataset-full
├── cat/
    ├── cat(keyboard)/
    ├── cat(sink)/ 
    ├── ... 
├── dog/
    ├── dog(surfboard) 
    ├── dog(boat)/ 
    ├── ...
├── bus/ 
├── ...
```

Beyond the generated MetaShift dataset, the scipt also genervates the meta-graphs for each class in `dataset/meta-graphs`. 
```plain
.
├── README.md
├── dataset/
    ├── generate_full_MetaShift.py
    ├── meta-graphs/             (generated meta-graph visualization) 
        ├──  cat_graph.jpg
        ├──  dog_graph.jpg
        ├──  ...
    ├── ...         
```

## Bonus: Generate the MetaShift-Attributes Dataset (subsets defined by subject attributes)

<p align='center'>
  <img width='100%' src='./docs/figures/MetaShift-Attributes-Examples.jpg'/>
<b>Figure: Example Subsets based on object attribute contexts. </b> the attribute is stated in parenthesis). MetaShift
covers attributes including activity (e.g., sitting, jumping), color (e.g., orange, white), material (e.g., wooden,
metallic), shape (e.g., round, square), and so on.
</p>

### Understanding `dataset/attributes_MetaShift/attributes-candidate-subsets.pkl`
`dataset/attributes_MetaShift/attributes-candidate-subsets.pkl` stores the metadata for MetaShift-Attributes, where each subset is defined by the attribute of the subject, e.g. `cat(orange)`, `cat(white)`, `dog(sitting)`, `dog(jumping)`. 

`attributes-candidate-subsets.pkl` has the same data format as `full-candidate-subsets.pkl`. To facilitate understanding, we have provided a notebook `dataset/attributes_MetaShift/understanding_attributes-candidate-subsets-pkl.ipynb` to show how to extract information from it. 

Basically, the pickle file stores a `collections.defaultdict(set)` object, which contains *4,962* keys. Each key is a string of the subset name like `cat(orange)`, and the corresponding value is a list of the IDs of the images that belong to this subset. The image IDs can be used to retrieve the image files from the Visual Genome dataset that you just downloaded. 

### Understanding `dataset/attributes_MetaShift/structured-attributes-candidate-subsets.pkl`
`dataset/attributes_MetaShift/structured-attributes-candidate-subsets.pkl` is very similar to `dataset/attributes_MetaShift/attributes-candidate-subsets.pkl`, but stores the metadata in a more structured way. The pickle file stores a 3-level nested dictionary, with the following structure:

```plain
.
├── key: 'color'
    ├── key: 'cat'              
        ├── key: 'orange'
            ├── value: a list of image IDs
├── key: 'activity'
    ├── key: 'dog'              
        ├── key: 'sitting'
            ├── value: a list of image IDs
        ├── ...
```

See the full attrribute ontology in `ATTRIBUTE_CONTEXT_ONTOLOGY` in `dataset/Constants.py`

```python
ATTRIBUTE_CONTEXT_ONTOLOGY = {
 'darkness': ['dark', 'bright'],
 'dryness': ['wet', 'dry'],
 'colorful': ['colorful', 'shiny'],
 'leaf': ['leafy', 'bare'],
 'emotion': ['happy', 'calm'],
 'sports': ['baseball', 'tennis'],
 'flatness': ['flat', 'curved'],
 'lightness': ['light', 'heavy'],
 'gender': ['male', 'female'],
 'width': ['wide', 'narrow'],
 'depth': ['deep', 'shallow'],
 'hardness': ['hard', 'soft'],
 'cleanliness': ['clean', 'dirty'],
 'switch': ['on', 'off'],
 'thickness': ['thin', 'thick'],
 'openness': ['open', 'closed'],
 'height': ['tall', 'short'],
 'length': ['long', 'short'],
 'fullness': ['full', 'empty'],
 'age': ['young', 'old'],
 'size': ['large', 'small'],
 'pattern': ['checkered', 'striped', 'dress', 'dotted'],
 'shape': ['round', 'rectangular', 'triangular', 'square'],
 'activity': ['waiting', 'staring', 'drinking', 'playing', 'eating', 'cooking', 'resting', 
              'sleeping', 'posing', 'talking', 'looking down', 'looking up', 'driving', 
              'reading', 'brushing teeth', 'flying', 'surfing', 'skiing', 'hanging'],
 'pose': ['walking', 'standing', 'lying', 'sitting', 'running', 'jumping', 'crouching', 
            'bending', 'smiling', 'grazing'],
 'material': ['wood', 'plastic', 'metal', 'glass', 'leather', 'leather', 'porcelain', 
            'concrete', 'paper', 'stone', 'brick'],
 'color': ['white', 'red', 'black', 'green', 'silver', 'gold', 'khaki', 'gray', 
            'dark', 'pink', 'dark blue', 'dark brown',
            'blue', 'yellow', 'tan', 'brown', 'orange', 'purple', 'beige', 'blond', 
            'brunette', 'maroon', 'light blue', 'light brown']
}
```



## Section 4.2: Evaluating Subpopulation Shifts
Run the python script `dataset/subpopulation_shift_cat_dog_indoor_outdoor.py` to reproduce the MetaShift subpopulation shift dataset (based on Visual Genome images) in the paper. 
```sh
cd dataset/
python subpopulation_shift_cat_dog_indoor_outdoor.py
```
The python script generates a “Cat vs. Dog” dataset, where the general contexts “indoor/outdoor” have a natural spurious correlation with the class labels. 


The following files will be generated by executing the python script `dataset/subpopulation_shift_cat_dog_indoor_outdoor.py`. 

### Output files (mixed version: for reproducing experiments)

```plain
/data/MetaShift/MetaShift-subpopulation-shift
├── imageID_to_group.pkl
├── train/
    ├── cat/             (more cat(indoor) images than cat(outdoor))
    ├── dog/             (more dog(outdoor) images than cat(indoor)) 
├── val_out_of_domain/
    ├── cat/             (cat(indoor):cat(outdoor)=1:1)
    ├── dog/             (dog(indoor):dog(outdoor)=1:1) 
```
where `imageID_to_group.pkl` is a dictionary with 4 keys : 
`'cat(outdoor)'`, `'cat(outdoor)'`, `'dog(outdoor)'`, `'dog(outdoor)'`. 
The corresponding value of each key is the list of the names of the images that belongs to that subset. 
Modify the global varaible `SUBPOPULATION_SHIFT_DATASET_FOLDER` to change the destination folder. 
You can tune the `NUM_MINORITY_IMG` to control the amount of subpopulation shift.  

### Output files (unmixed version, for other potential uses)
To facilitate other potential uses, we also outputs an unmixed version, where we output the `'cat(outdoor)'`, `'cat(outdoor)'`, `'dog(outdoor)'`, `'dog(outdoor)'` into 4 seperate folders. 
Modify the global varaible `CUSTOM_SPLIT_DATASET_FOLDER` to change the destination folder. 
```plain
/data/MetaShift/MetaShift-Cat-Dog-indoor-outdoor
├── imageID_to_group.pkl
├── train/
    ├── cat/             (all cat(indoor) images)
        ├── cat(indoor)/
    ├── dog/             (all dog(outdoor) images) 
        ├── dog(outdoor)/
├── test/
    ├── cat/             (all cat(outdoor) images)
        ├── cat(outdoor)/
    ├── dog/             (all dog(indoor) images) 
        ├── dog(indoor)/
```


## Appendix D: Constructing MetaShift from COCO Dataset
The notebook `dataset/extend_to_COCO/coco_MetaShift.ipynb` reproduces the COCO subpopulation shift dataset in paper Appendix D. Executing the notebook would construct a “Cat vs. Dog” task based on COCO images, where the “indoor/outdoor” contexts are spuriously correlated with the class labels. 

### Install COCO Dependencies
Install pycocotools (for evaluation on COCO):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### COCO Data preparation

[2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

[2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
/home/ubuntu/data/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
Modify the global varaible `IMAGE_DATA_FOLDER` to change the COCO image folder. 

### Output files (mixed version: for reproducing experiments)

The following files will be generated by executing the notebook. 
```plain
/data/MetaShift/COCO-Cat-Dog-indoor-outdoor
├── imageID_to_group.pkl
├── train/
    ├── cat/
    ├── dog/ 
├── val_out_of_domain/
    ├── cat/
    ├── dog/ 
```
where `imageID_to_group.pkl` is a dictionary with 4 keys : 
`'cat(outdoor)'`, `'cat(outdoor)'`, `'dog(outdoor)'`, `'dog(outdoor)'`. 
The corresponding value of each key is the list of the names of the images that belongs to that subset. Modify the global varaible `CUSTOM_SPLIT_DATASET_FOLDER` to change the destination folder. 



## Section 4.1: Evaluating Domain Generalization
Run the python script `dataset/domain_generalization_cat_dog.py` to reproduce the MetaShift domain generalization dataset (based on Visual Genome images) in the paper.
```sh
cd dataset/
python domain_generalization_cat_dog.py
```

### Output files (cat vs. dog, unmixed version)
The following files will be generated by executing the python script `dataset/domain_generalization_cat_dog.py`. Modify the global varaible `CUSTOM_SPLIT_DATASET_FOLDER` to change the COCO image folder. 
```plain
/data/MetaShift/Domain-Generalization-Cat-Dog
├── train/
    ├── cat/
        ├── cat(sofa)/              (The cat training data is always cat(\emph{sofa + bed}) ) 
        ├── cat(bed)/               (The cat training data is always cat(\emph{sofa + bed}) )
    ├── dog/
        ├── dog(cabinet)/           (Experiment 1: the dog training data is dog(\emph{cabinet + bed}))
        ├── dog(bed)/               (Experiment 1: the dog training data is dog(\emph{cabinet + bed}))

        ├── dog(bag)/               (Experiment 2: the dog training data is dog(\emph{bag + box}))
        ├── dog(box)/               (Experiment 2: the dog training data is dog(\emph{bag + box}))

        ├── dog(bench)/             (Experiment 3: the dog training data is dog(\emph{bench + bike}))
        ├── dog(bike)/              (Experiment 3: the dog training data is dog(\emph{bench + bike}))

        ├── dog(boat)/              (Experiment 4: the dog training data is dog(\emph{boat + surfboard}))
        ├── dog(surfboard)/         (Experiment 4: the dog training data is dog(\emph{boat + surfboard}))

├── test/
    ├── dog/
        ├── dog(shelf)/             (The test set we used in the paper)
        ├── dog(sofa)/             
        ├── dog(grass)/             
        ├── dog(vehicle)/             
        ├── dog(cap)/                         
    ├── cat/
        ├── cat(shelf)/
        ├── cat(grass)/
        ├── cat(sink)/
        ├── cat(computer)/
        ├── cat(box)/
        ├── cat(book)/
```


## Code for Distribution Shift Experiments
The python script `experiments/distribution_shift/main_generalization.py` is the entry point for running the distribution shift experiemnts for Section 4.2 (Evaluating Subpopulation Shifts) and Appendix D (Constructing MetaShift from COCO Dataset), and Section 4.1 (Evaluating Domain Generalization). As a running example, the default value for `--data` in `argparse` is `/data/MetaShift/MetaShift-subpopulation-shift` (i.e., for Section 4.2). 


```sh
clear && CUDA_VISIBLE_DEVICES=3 python main_generalization.py --num-domains 2 --algorithm ERM 
clear && CUDA_VISIBLE_DEVICES=4 python main_generalization.py --num-domains 2 --algorithm GroupDRO 
clear && CUDA_VISIBLE_DEVICES=5 python main_generalization.py --num-domains 2 --algorithm IRM 
clear && CUDA_VISIBLE_DEVICES=6 python main_generalization.py --num-domains 2 --algorithm CORAL 
clear && CUDA_VISIBLE_DEVICES=7 python main_generalization.py --num-domains 2 --algorithm CDANN 
```

Our code is based on the [DomainBed](https://github.com/facebookresearch/DomainBed), as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). The codebase also provides [many additional algorithms](experiments/subpopulation_shift/algorithms.py). Many thanks to the authors and developers! 