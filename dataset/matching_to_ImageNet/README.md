# Match MetaShift with ImageNet 

`match_metashift_imagenet.py` walks through the following things:

- matching classes in MetaShift with ImageNet classes
- select the matched classes and generate the subsets from the full version 
- generate `selected-candidate-subsets.pkl`, which contains the classes and subsets information of the ImageNet-1k matched version MetaShift.
- check the coverage of the selected classes in ImageNet-1k

**Notes:**

- use [wordnet](https://www.nltk.org/howto/wordnet.html) to do the matching:
  ```python
    ! pip install nltk
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
  ```


**Matching Result:**
  
  - classes in MetaShift matched to ImageNet-1k: 427
  - classes selected in matched version: 216
  - ImageNet-1k classes covered in matched version: 867
