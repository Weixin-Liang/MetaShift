"""
Generate MetaDataset with train/test split 

"""

CUSTOM_SPLIT_DATASET_FOLDER = '/data/MetaShift/Domain-Generalization-Cat-Dog'

import pandas as pd 
import seaborn as sns

import pickle
import numpy as np
import json, re, math
from collections import Counter, defaultdict
from itertools import repeat
import pprint
import os, errno
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil # for copy files
import networkx as nx # graph vis
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import Constants
IMAGE_DATA_FOLDER          = Constants.IMAGE_DATA_FOLDER

from generate_full_MetaShift import preprocess_groups, build_subset_graph, copy_image_for_subject


def print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str):
    ##################################
    # Community detection 
    ##################################
    G = build_subset_graph(subject_data, node_name_to_img_id, trainsg_dupes, subject_str)

    import networkx.algorithms.community as nxcom

    # Find the communities
    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    # Count the communities
    print(f"The graph has {len(communities)} communities.")
    for community in communities:
        community_merged = set()
        for node_str in community:
            node_str = node_str.replace('\n', '')
            node_image_IDs = node_name_to_img_id[node_str]
            community_merged.update(node_image_IDs)
            # print(node_str , len(node_image_IDs), end=';')

        print('total size:',len(community_merged))
        community_set = set([ x.replace('\n', '') for x in community])
        print(community_set, '\n\n')
    return G 



def parse_dataset_scheme(dataset_scheme, node_name_to_img_id, exclude_img_id=set(), split='test', copy=True):
    """
    exclude_img_id contains both trainsg_dupes and test images that we do not want to leak 
    """
    community_name_to_img_id = defaultdict(set)
    all_img_id = set()

    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                community_name_to_img_id[community_name].update(node_name_to_img_id[node_name] - exclude_img_id)
                all_img_id.update(node_name_to_img_id[node_name] - exclude_img_id)
            if copy:
                print(community_name, 'Size:', len(community_name_to_img_id[community_name]) )


        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        if copy:
            root_folder = os.path.join(CUSTOM_SPLIT_DATASET_FOLDER, split)
            copy_image_for_subject(root_folder, subject_str, dataset_scheme[subject_str], community_name_to_img_id, trainsg_dupes=set(), use_symlink=False) # use False to share 

    return community_name_to_img_id, all_img_id


def get_all_nodes_in_dataset(dataset_scheme):
    all_nodes = set()
    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                all_nodes.add(node_name)
    return all_nodes

def generate_splitted_metadaset():

    if os.path.isdir(CUSTOM_SPLIT_DATASET_FOLDER): 
        shutil.rmtree(CUSTOM_SPLIT_DATASET_FOLDER) 
    os.makedirs(CUSTOM_SPLIT_DATASET_FOLDER, exist_ok = False)


    node_name_to_img_id, most_common_list, subjects_to_all_set, subject_group_summary_dict = preprocess_groups(output_files_flag=False)

    ##################################
    # Removing ambiguous images that have both cats and dogs 
    ##################################
    trainsg_dupes = node_name_to_img_id['cat(dog)'] # can also use 'dog(cat)'
    subject_str_to_Graphs = dict()


    for subject_str in ['cat', 'dog']:
        subject_data = [ x for x in subject_group_summary_dict[subject_str].keys() if x not in ['cat(dog)', 'dog(cat)'] ]
        # print('subject_data', subject_data)
        ##################################
        # Print detected communities in Meta-Graph
        ##################################
        G = print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str) # print detected communities, which guides us the train/test split. 
        subject_str_to_Graphs[subject_str] = G




    train_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'cat': {
            # The cat training data is always cat(\emph{sofa + bed}) 
            'cat(sofa)': {'cat(cup)', 'cat(sofa)', 'cat(chair)'},
            'cat(bed)':  {'cat(bed)', 'cat(comforter)', 'cat(sheet)', 'cat(blanket)', 'cat(remote control)', 'cat(pillow)', 'cat(couch)'},
        }, 
        'dog': {
            # Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            'dog(cabinet)': {'dog(floor)', 'dog(clothes)', 'dog(towel)', 'dog(door)', 'dog(rug)', 'dog(cabinet)'}, 
            'dog(bed)': {'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(pillow)', 'dog(lamp)', 'dog(couch)', 'dog(books)', 'dog(curtain)'}, 

            # Experiment 2: the dog training data is dog(\emph{bag + box}), and its distance to dog(\emph{shelf}) is $d$=0.71. 
            'dog(bag)': {'dog(bag)', 'dog(backpack)', 'dog(purse)'},
            'dog(box)': {'dog(box)', 'dog(container)', 'dog(food)', 'dog(table)', 'dog(plate)', 'dog(cup)'} ,

            # Experiment 3: the dog training data is dog(\emph{bench + bike}) with distance $d$=1.12
            'dog(bench)': {'dog(bench)', 'dog(trash can)'} ,
            'dog(bike)': {'dog(basket)', 'dog(woman)', 'dog(bike)', 'dog(bicycle)'},

            # Experiment 4: the dog training data is dog(\emph{boat + surfboard}) with distance $d$=1.43.   
            'dog(boat)': {'dog(frisbee)', 'dog(rope)', 'dog(flag)', 'dog(trees)', 'dog(boat)'},
            'dog(surfboard)': {'dog(water)', 'dog(surfboard)', 'dog(sand)'}, # 'dog(ball)', 
        }
    }

    test_set_scheme = {
        'cat': {
            'cat(shelf)': {'cat(container)', 'cat(shelf)', 'cat(vase)', 'cat(bowl)'},
        },
        'dog': {
            # In MetaDataset paper, the test images are all dogs. However, for completeness, we also provide cat images here. 
            'dog(shelf)': {'dog(desk)', 'dog(screen)', 'dog(laptop)', 'dog(shelf)', 'dog(picture)', 'dog(chair)'}, 
        },
    }

    additional_test_set_scheme = {
        'cat': {
            'cat(grass)': {'cat(house)', 'cat(car)', 'cat(grass)', 'cat(bird)'},
            'cat(sink)': {'cat(sink)', 'cat(bottle)', 'cat(faucet)', 'cat(towel)', 'cat(toilet)'}, 
            'cat(computer)': {'cat(speaker)', 'cat(computer)', 'cat(screen)', 'cat(laptop)', 'cat(computer mouse)', 'cat(keyboard)', 'cat(monitor)', 'cat(desk)',}, 
            'cat(box)': {'cat(box)', 'cat(paper)', 'cat(suitcase)', 'cat(bag)',}, 
            'cat(book)': {'cat(books)', 'cat(book)', 'cat(television)', 'cat(bookshelf)', 'cat(blinds)',},
        },
        'dog': {
            'dog(sofa)': {'dog(sofa)', 'dog(television)', 'dog(carpet)',  'dog(phone)', 'dog(book)',}, 
            'dog(grass)': {'dog(house)', 'dog(grass)', 'dog(horse)', 'dog(cow)', 'dog(sheep)','dog(animal)'}, 
            'dog(vehicle)': {'dog(car)', 'dog(motorcycle)', 'dog(truck)', 'dog(bike)', 'dog(basket)', 'dog(bicycle)', 'dog(skateboard)', }, 
            'dog(cap)': {'dog(cap)', 'dog(scarf)', 'dog(jacket)', 'dog(toy)', 'dog(collar)', 'dog(tie)'},
        },
    }


    print('========== test set info ==========')
    test_community_name_to_img_id, test_all_img_id = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test')
    # print('test_all_img_id', len(test_all_img_id))
    print('========== train set info ==========')
    train_community_name_to_img_id, train_all_img_id = parse_dataset_scheme(train_set_scheme, node_name_to_img_id, exclude_img_id=test_all_img_id.union(trainsg_dupes), split='train')
    print('========== additional test set info ==========')
    additional_test_community_name_to_img_id, additional_test_all_img_id = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=train_all_img_id.union(trainsg_dupes), split='test')


    ##################################
    # **Quantifying the distance between train and test subsets**
    # Please be advised that before making MetaShift public, 
    # we have made further efforts to reduce the label errors propagated from Visual Genome. 
    # Therefore, we expect a slight change in the exact experiment numbers.  
    ##################################
    
    print('========== Quantifying the distance between train and test subsets ==========')
    test_community_name_to_img_id, _ = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    train_community_name_to_img_id, _ = parse_dataset_scheme(train_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='train', copy=False)
    additional_test_community_name_to_img_id, _ = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test')

    community_name_to_img_id = test_community_name_to_img_id.copy()
    community_name_to_img_id.update(train_community_name_to_img_id)
    community_name_to_img_id.update(additional_test_community_name_to_img_id)
    dog_community_name_list = sorted(train_set_scheme['dog']) + sorted(test_set_scheme['dog']) + sorted(additional_test_set_scheme['dog'])
    
    G = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None)

    spectral_pos = nx.spectral_layout(
        G=G, 
        dim=5,
        )
    
    for subset_A, subset_B in [
        ['dog(cabinet)', 'dog(bed)'],
        ['dog(bag)', 'dog(box)'],
        ['dog(bench)', 'dog(bike)'],
        ['dog(boat)', 'dog(surfboard)'],
    ]:
        distance_A = np.linalg.norm(spectral_pos[subset_A.replace('(', '\n(')] - spectral_pos['dog\n(shelf)'])
        distance_B = np.linalg.norm(spectral_pos[subset_B.replace('(', '\n(')] - spectral_pos['dog\n(shelf)'])
        
        print('Distance from {}+{} to {}: {}'.format(
            subset_A, subset_B, 'dog(shelf)', 
            0.5 * (distance_A + distance_B)
            )
        )

        
    return

if __name__ == '__main__':
    generate_splitted_metadaset()

