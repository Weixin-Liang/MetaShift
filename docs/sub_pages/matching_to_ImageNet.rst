Matching MetaShift with ImageNet 
============================================

Welcome! This is the project website of our paper: `MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts <https://openreview.net/forum?id=MTex8qKavoS>`__ (ICLR 2022). 
`[PDF] <https://arxiv.org/abs/2202.06523>`__
`[Video] <https://recorder-v3.slideslive.com/#/share?share=64243&s=4b8a00e2-83f3-4775-879f-70de42374ec6>`__
`[Slides] <https://drive.google.com/file/d/1PDQSrNQWAJL_cx-KpV1CchUJwk2MgPFC/view?usp=sharing>`__

*Contributed Talk at* `ICML 2022 Workshop on Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet <https://shift-happens-benchmark.github.io/>`_

Given that MetaShift is a flexible framework to generate a large number of 
real-world distribution shifts that are well-annotated and controlled, 
we can use it to construct a new dataset of specific classes and subpopulations.

Introduction of ImageNet-1k
--------------------------------------------------
ImageNet is an image database organized according to the WordNet hierarchy, 
in which each node of the hierarchy is depicted by hundreds and thousands of images. 
The full ImageNet contains 60,942 nodes, while the 1,000 ImageNet classes contains 
only 2,155 nodes. 
(Refer to `ImageNet-1k class hierarchy <https://observablehq.com/@mbostock/imagenet-hierarchy>`_)



Matching Method
--------------------------------------------------
We use `wordnet <https://www.nltk.org/howto/wordnet.html>`_ to do the matching.

For each meta-data tag of the classes and the subsets of the context as well as the attributes, 
we search in the ImageNet-1k hierarchy to find if it has the label with the same wordnet id. 
The meta-data tag in MetaShift may represent a greater domain than the leaf nodes of the ImageNet hierarchy, 
for example, MetaShift has only one general *"cat"* class, while the ImageNet *"domestic cat"* and *"wildcat"* 
under the *"cat"* hierarchy, and each kind of cat also has several different breeds. 
In the matching procedure, all breeds under *"cat"* hierarchy will be matched to *"cat"* class in MetaShift.

**Check Coverage**

To verify the coverage of the generated dataset over the ImageNet-1k, we count in the 
following methodology: for each meta-data of the matched version of MetaShift, we locate the tags 
in the ImageNet hierarchy. If it is a non-leaf node, then mark all of its leaf nodes, otherwise mark the 
leaf node itself. 

 


Matching Result
--------------------------------------------------
Originally, the MetaShift contains a collection of *12,868* sets of natural images across *410* classes. 
After matching, we selected *5,040* sets of images across *261* classes, where each tag of it can 
be found in the ImageNet-1k dataset. The matched version covers *867* out of 1,000 classes in ImageNet-1k. 
Each class in the ImageNet-matched MetaShift contains *2301.6* images on average, and *19.3* subsets capturing images in different contexts.  
The unmatched portions of our datasets can be potentially used for OOD (out-of-distribution) detection, and we will delay it in future work.

.. figure:: https://user-images.githubusercontent.com/67904087/179450229-8ed0bfdd-ea0b-404d-9c34-659e2a96d7d8.png
   :width: 100 %
   :align: center
   :alt: 

Task Construction
--------------------------------------------------
The 261 classes over 5,040 sets of images provide enumerable options for task construction.    

Binary classification Task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can select two classes of the dataset to construct binary classification task. Here we represent a method to construct the tasks with the MetaShift:
We first filter the classes whose subsets are less than a threshold. For the selected classes, we find the common parent nodes of two classes in the ImageNet hierarchy, which can be used to evaluate their similarities. 
To be specific, if we use *5* as the subsets filtering threshold, and select the pairs of classes who have common parent nodes in the second hierarchy of the ImageNet, we can get *19,024* binary classification tasks as a result. 

Multiclass Classification Task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The context can have a great impact on the classification because of the difference between it in train and test set. 
In Table 1, we select 5 classes to do evaluation on 3 pre-trained ImageNet models: ResNet18, ResNet50 and VGG 16. 
The accuracy varies drastically across different classes depending on the distribution shifts of the class. 
The classification accuracy of elephant is relatively high since the contexts of elephant images are mostly outdoor 
in both ImageNet and MetaShift. In contrast, the contexts of cat images varies a lot. 
The subsets contain both indoor and outdoor contexts in MetaShift, such as toilet, grass and other heterogeneous contexts, 
which poses great distribution shifts. The lower accuracy of cat classification indicates the ImageNet models' incapability in handling distribution shift.

*Code released in* `shift-happens-benchmark from icml workshop <https://github.com/shift-happens-benchmark/icml-2022>`_ 

.. figure:: https://user-images.githubusercontent.com/67904087/179453087-c3898dc6-cd32-41c9-909b-fa0b69e1fcae.png
   :width: 100 %
   :align: center
   :alt: 

Citation
--------

.. code-block:: bibtex

   @InProceedings{liang2022metashift,
   title={MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts},
   author={Weixin Liang and James Zou},
   booktitle={International Conference on Learning Representations},
   year={2022},
   url={https://openreview.net/forum?id=MTex8qKavoS}
   }