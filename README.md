# CLIP-Image-Classification



### Use

```python
!git clone https://github.com/aicrumb/CLIP-Image-Classification # if not in a notebook, run in console (w/o the "!")
import os
os.chdir("CLIP-Image-Classification")

from classify import load, classify, encode

filename = "../input.jpg"

load_categories = "imagenet"

print("loading categories")
load(load_categories)

print("classifying")
print(classify(filename))
```



#### `load`

```python
load("imagenet") #imagenet categories
load("pokemon") #loads a list of 721 pokemon names as categories
load("dog vs cat") #dog and cat as categories
load("emojis") #emojis :)
load(["banana", "elephant", "monkey"]) #any custom words in a list will do as well
```

#### `classify`
```python
classify(filename) #returns the highest scoring class
classify(filename, return_raw=True) #returns the scores for all the classes (cosine_similarity)
```

#### `encode`

this will return CLIP's raw encoding of an image or text if you need it.

```python
encode("input.jpg") #encode based on filename, it'll be detected if it ends w/ png, jpg, or jpeg
encode("an image of a flower") #encode based on text
```



### Examples

Classify an image: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kingchloexx/CLIP-Image-Classification/blob/main/Multi_Domain_Pretrained_Classifier_with_CLIP.ipynb)

Search within an image: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kingchloexx/CLIP-Image-Classification/blob/main/Clip_Search.ipynb)

### Dependencies

I made it as simple as I could lmao, first you've gotta install the dependencies which can be done with this block of text if you're using Google Colab:

```python
import subprocess

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = ""
else:
    torch_version_suffix = "+cu110"
!git clone https://github.com/kingchloexx/CLIP-Image-Classification
! pip install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html ftfy regex
!pip install ftfy

```

If you're using a conda environment (outside of google colab), make sure you have an nvidea graphics card, once you've `conda activate`ed your environment, use `conda install cudatoolkit` and at the time of writing this, the command for installing the correct torch version would be

```
pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html ftfy regex
```

if you try the script and it says there are dependencies missing, usually a `pip install [dependency name]` will fix it

👍 - [Chloe](https://github.com/aicrumb)
