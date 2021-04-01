import torch
import numpy as np
from PIL import Image
import os
import random
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
import subprocess
InteractiveShell.ast_node_interactivity = "all"
import glob
import clip
perceptor, preprocess = clip.load('ViT-B/32')
import sys
c_encs=[]
categories = []
def load(categorylist):
    global c_encs
    global categories
    load_categories = categorylist #@param ["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]
    if(load_categories not in ["emojis", "imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]):
        categories = categorylist
    elif(load_categories=="imagenet"):
        import pandas as pd
        categories = pd.read_csv("categories/map_clsloc.txt", sep=" ", header = None)[2]
        for category in range(len(categories)):
            categories[category] = categories[category].replace("_", " ")
    elif(load_categories=="dog vs cat"):
        categories = ["dog", "cat"]
    elif(load_categories=="pokemon"):
        import pandas as pd
        categories = pd.read_csv("categories/pokemon.txt", sep=".", header=None)[1]
    elif(load_categories=="words in the communist manifesto"):
        ccc = open("categories/communism.txt", "r").read().split()
        categories = []
        for i in ccc:
            if i not in categories:
                categories.append(i)
    elif(load_categories=="emojis"):
        categories = open("categories/emojis.txt", "r").readlines()
    c_encs = [perceptor.encode_text(clip.tokenize(category).cuda()).detach().clone() for category in categories]

import PIL
def classify(filename, return_raw=False):
    im_enc = perceptor.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to("cpu"))
    distances = [torch.cosine_similarity(e, im_enc).item() for e in c_encs]

    if(return_raw==False):
        return categories[int(distances.index(max(distances)))]
    else:
        return distances
def encode(object):
    o = object.lower()
    if("jpg" in o[-5:]) or ("png" in o[-5:]) or ("jpeg" in o[-5:]):
        return perceptor.encode_image(preprocess(Image.open(object)).unsqueeze(0).to("cpu"))
    else:
        # o = object.lower()
        return perceptor.encode_text(clip.tokenize(object).cuda()).detach().clone()
