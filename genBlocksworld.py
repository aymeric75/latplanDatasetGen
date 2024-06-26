import numpy as np
import os.path
import itertools
import torch
from torchvision import datasets, transforms
#from ..util.tuning import parameters
#from .common import *
#from .normalization import normalize_transitions, normalize_transitions_objects
#from ..puzzles.objutil import bboxes_to_coord, random_object_masking
#from . import common
#from ..util.stacktrace import format

import matplotlib.pyplot as plt

import sys
#sys.path.append(r"..latplanClonedLossTerm/latplan")
sys.path.insert(0, '/workspace/latplanClonedEnforce')
import latplan
from latplan.puzzles.objutil import image_to_tiled_objects, tiled_bboxes
from latplan.main.normalization import normalize_transitions, normalize_transitions_objects
sys.path.remove('/workspace/latplanClonedEnforce')
import torch.nn.functional as F

# [[1, 3], 1]
def flatten(nested_list):
    """Flatten a nested list of lists into a single list, handling nested lists at any level."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(item)  # Recursively flatten each sublist
        else:
            flat_list.append(item)
    return flat_list


#################### UTIL FUNCTIONS


# Convert to grayscale using the luminosity method
def rgb_to_grayscale(rgb_images):
    grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07])
    return np.stack((grayscale,)*3, axis=-1)




def plot_image(a,name):
    #plt.figure(figsize=(6,6))
    plt.figure()
    #plt.imshow(a,interpolation='nearest',cmap='gray',)
    plt.imshow(a)
    plt.savefig(name)



def normalize_here(x, save=True):

    print("HYYYYHHHHHH")
    print(x.shape) # (40000, 30, 45, 3)

    mean               = np.mean(x,axis=0)

    print(mean.shape) # (30, 45, 3)

    std                = np.std(x,axis=0)
    # if save:
    #     parameters["mean"] = [mean.tolist()]
    #     parameters["std"]  = [std.tolist()]
    print("normalized shape:",mean.shape,std.shape)
    return (x - mean)/(std+1e-20), mean, std


def normalize_transitions_here(pres,sucs):
    """Normalize a dataset for image-based input format.
Normalization is performed across batches."""
    print("IN normalize_transitions HERE")
    B, *F = pres.shape


    transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
    print(transitions.shape) # (50, 2, 30, 45, 3)
    print(np.reshape(transitions, [-1, *F]).shape) # (100, 30, 45, 3)
    

    normalized, mean, std = normalize_here(np.reshape(transitions, [-1, *F])) # [2BO, F]
    print("normies")
    #print(normalized[:4])

    states      = normalized.reshape([-1, *F])
    transitions = states.reshape([-1, 2, *F])
    return transitions, states, mean, std


#### ALL POSSIBLE "CLEAR" combinations
def all_clear_combination():
    numbers = [0, 1, 2, 3]
    all_combinations = []
    for r in range(1, len(numbers) + 1):
        combinations = itertools.combinations(numbers, r)
        all_combinations.extend(combinations)
    return all_combinations


########################################
numbers = [0, 1, 2]

#### TOWERs combinations
def return_all_possible_towers():
    selected_permutations = []
    for r in range(2, 4):
        permutations = itertools.permutations(numbers, r)
        selected_permutations.extend(permutations)
    return selected_permutations


def return_pairs_of_above_from_a_tower(tower):
    # where tower is like [3 0 1] i.e. block 3 on top of block 0 on top of block 1
    assert len(tower) > 1
    pairs = []
    for i, b in enumerate(tower):
        for j in range(i+1, len(tower)):
            pair = [b, tower[j]]
            pairs.append(pair)
    return pairs


def return_all_towers_and_xAboveY():
    all_towers = return_all_possible_towers()
    all_towers_xAboveY = {} # each ele is a list of list
    # like [[0, 1], [0, 2], [1, 2]] i.e. 0 on top of 1 on top of 2, 1 on top of 2
    for t in all_towers:
        all_towers_xAboveY[str(t)] = return_pairs_of_above_from_a_tower(t)
    return all_towers_xAboveY



# exceptions can be like [[1, 3, 2]]
def item_has_nothing_on_left_except(lefts_list, number_or_tower, exceptions=None):
   


    retour = True
    for i, l in enumerate(lefts_list):
        if type(number_or_tower) is list and len(number_or_tower) > 0:
            reference_block = number_or_tower[0]
        else:
            reference_block = number_or_tower
        
        if exceptions==None or (not exceptions):
            #if number_or_tower == 2: print("i: {}, l: {}".format(str(i), str(l)))
            if i == reference_block and l:
                #if number_or_tower == 2: print("was heere")
                retour = False
                break
        else:
            if i == reference_block and l:
                flatten_exceptions = flatten(exceptions)
                for el in l:
                    if el not in flatten_exceptions:
                        retour = False
                        break
        
    return retour


# all combinations of towers with 2 to 4 numbers
#     then for each, retrieve the remaining


import numpy as np

def fromTowerAndLeftsBuildState(tower, lefts,thecount):
    
    #  SHOULD RETURN [2  1  0]
    final_state = []
    if type(tower) is list:
        if tower and len(tower) ==  2:
            if type(tower[0]) is list and type(tower[1]) is list:
                    if lefts[tower[0][0]]:
                        final_state.append(tower[1])
                        final_state.append(tower[0])
                    else:
                        final_state.append(tower[0])
                        final_state.append(tower[1])
                    return final_state
    

    # 
    remaining_elements_of_state = []
    for i in range(3):
        if i not in tower:
            remaining_elements_of_state.append(i)
    if tower:     
        remaining_elements_of_state.append(tower)


    present_elements = remaining_elements_of_state.copy()
    placed_elements = []
    counter=0

    while(len(final_state) != len(present_elements)):
        #print("counter : {}".format(str(counter)))
        #print(remaining_elements_of_state)
        for el in remaining_elements_of_state:

            if el not in final_state and item_has_nothing_on_left_except(lefts, el, exceptions=placed_elements):
                final_state.append(el)
                placed_elements.append(el)
                remaining_elements_of_state.remove(el)
            
            # if counter > 4:
            #     print("on es la")
            #     print(final_state)
            #     print(placed_elements)
            #     print(remaining_elements_of_state)
            #     print(present_elements)
            if counter > 100:
                print("thecount {}".format(str(thecount)))
                print(tower)
                print(lefts)
                return False
                
            counter+=1

    return final_state






# for c in all_clear_combination():
# all combi, in any order of from 2 to 4 numbers among 0, 1, 2, 3
#    then for each of them (which constitutes a tower)
#              e.g. [3 0 1]  (= from top to bottom) ==> 
#  


#### TOWERs combinations
def return_all_possible_towers_by_size(size):
    permutations = itertools.permutations([0, 1, 2], size)
    return list(permutations)


all_states = []

# NO tower
all_states_no_tower = list(itertools.permutations([0, 1, 2]))
for e in all_states_no_tower:
    all_states.append(list(e))

# ONE TOWER of size two
for t in return_all_possible_towers_by_size(2):

    eles = [0,1,2]
    new_state = []
    new_state.append(list(t))
    for e in eles:
        if e not in list(t):
            new_state.append(e)

    new_states = list(itertools.permutations(new_state))

    all_states.extend(new_states)



# ONE TOWER of size 3
for t in return_all_possible_towers_by_size(3):

    eles = [0,1,2]
    new_state = []
    new_state.append(list(t))
    for e in eles:
        if e not in list(t):
            new_state.append(e)

    new_states = list(itertools.permutations(new_state))

    all_states.extend(new_states)

# Two towers (when 4 blocks !!)

# two_towers = [ [(0, 1), (2, 3)],
#     [(0, 2), (1, 3)],
#     [(0, 3), (1, 2)]
#     ]

# for two in two_towers:
#     all_states.append([list(two[0]), list(two[1])])
#     all_states.append([list(two[1]), list(two[0])])


print(all_states[:10]) # [[0, 1, 2], [0, 2, 1]]
# [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], ([0, 1], 2), (2, [0, 1]), ([0, 2], 1), (1, [0, 2])]

# ([0, 1], 2)  == tour of a 0 and a 1 THEN the 2 on the right

print(len(all_states))


# 3 colored blocks

# NBER OF MOVES PER STATE
#    when all on table: 6 moves, 

#    when 1 tower of 2: 5 moves

#     1 tower of 3: 2

#   NBER OF STATEs
#        #combinations of 3 blocks on table: 6

#        #combinations of 2 blocks among 3 then by 2: 6 x 2

#        # combination of 3 blocks: 6

#    number of total moves:
#                          6    x   6
#                       +  12   x   5
#                       + 6  x 2 
#           108 !






def return_top_element(tower):
    if type(tower) is list:
        return tower[0]
    else:
        return tower



# state like [[1, 0], 2, 3]
def return_cleared_and_not_cleared_blocks(state):
    liste_clears = []
    liste_not_clears = []
    for s in state:
        liste_clears.append(return_top_element(s))
    for b in [0,1,2]:
        if b not in liste_clears:
            liste_not_clears.append(b)
    return liste_clears, liste_not_clears


def block_on_top(block, state):
    for s in state:
        if type(s) is list:
            if block in s:
                if block != s[0]:
                    index_block_on_top = s.index(block) - 1
                    return s[index_block_on_top]
    return


def valid_FromAllTableToAllTable(state1, state2):
    
    if not (len(state1) == 3 and len(state2) == 3):
        return False
    if state1 == state2:
        return False

    s11, s12, s13 = state1

    allowed_moves = [

        [s12, s11, s13],
        [s12, s13, s11],
        [s11, s13, s12],
        [s13, s11, s12],
    ]
    
    
    if state2 not in allowed_moves:
        return False

    return True




def valid_FromAllTableToOneTower(state1, state2):

    if not (len(state1)==3 and len(state2) == 2):
        return False
    if state1 == state2:
        return False
    
    s11, s12, s13 = state1


    

    allowed_moves = [
        [[s11, s12], s13], [s12, [s11, s13]]
    ,
    # moving s12
    
        [[s12, s11], s13], [s11, [s12, s13]]
    ,
    # moving s13
    
        [[s13, s11], s12], [s11, [s13, s12]]
    ]
 
    if list(state2) not in allowed_moves:
        return False

    return True

def valid_FromOneTowerOfTwoToAllOnTable(state1, state2):


    if not (len(state1) == 2 and len(state2) == 3):
        return False
    if state1 == state2:
        return False
    
    el11, el12 = state1

    if type(el11) is list:
        allowed_moves = [
            [el11[0], el11[1], el12],
            [el11[1], el11[0], el12],
            [el11[1], el12, el11[0]]
        ]
        if list(state2) not in allowed_moves:
            return False


    if type(el12) is list:
        allowed_moves = [
            [el11, el12[1], el12[0]],
            [el11, el12[0], el12[1]],
            [el12[0], el11, el12[1]]
        ]
        if list(state2) not in allowed_moves:
            return False

    return True



def valid_FromOneTowerOfTwoToAnotherTowerOfTwo(state1, state2):

    if not (len(state1)==2 and len(state2) == 2):
        return False
    if state1 == state2:
        return False    
    # 

    el11, el12 = state1
    el21, el22 = state2

    if type(el11) is list:
        allowed_moves = [
            [el11[1], [el11[0], el12]],
        ]
        if list(state2) not in allowed_moves:
            return False

    elif type(el12) is list:
        allowed_moves = [
            [[el12[0], el11], el12[1]],
        ]
        if list(state2) not in allowed_moves:
            return False

    return True


# # ([0, 2], 1), ([1, 0, 2],)
def valid_FromOneTowerOfTwoToOneTowerOfThree(state1, state2):

    if not (len(state1)==2 and len(state2) == 1):
        return False
    if state1 == state2:
        return False    

    el11, el12 = state1

    if type(el11) is list:
        allowed_moves = [
            [[el12, el11[0], el11[1]]]
        ]
        # print("state2")
        # print(list(state2))
        # print(allowed_moves)
        if list(state2) not in allowed_moves:
            return False

    elif type(el12) is list:
        allowed_moves = [
            [[el11, el12[0], el12[1]]]
        ]
        if list(state2) not in allowed_moves:
            return False
    return True



def valid_FromOneTowerOfThree(state1, state2):

    if not (len(state1)==1 and len(state2) == 2):
        return False
    if state1 == state2:
        return False
    
    el11 = state1

    allowed_moves = [
        [el11[0][0], [el11[0][1], el11[0][2]]],
        [[el11[0][1], el11[0][2]], el11[0][0]]
    ]
    if list(state2) not in allowed_moves:
        return False
    return True




#   Possible moves (from a fixed state)
#    when all on table:
#                       TO all on table: 6
#                       TO one tower of 2: 6

#                   

#    when 1 tower of 2: 
#                       TO 3 on table: 2 moves
#                       TO another tower of 2: 1 move
#                       TO 1 tower of 3:  1 move
#                           ==> 4 MOVES total

#     1 tower of 3: 2 MOVES
#                   



def check_if_valid_transition_BIS(state1, state2):
    # nber_cleared_blocksState1 = len(state1)
    # nber_cleared_blocksState2 = len(state2)
    # if abs(nber_cleared_blocksState1 - nber_cleared_blocksState2) > 1:
    #     return False


    if (valid_FromAllTableToAllTable(state1, state2) or 
        valid_FromAllTableToOneTower(state1, state2) or 
        valid_FromOneTowerOfTwoToAllOnTable(state1, state2) or
        valid_FromOneTowerOfTwoToAnotherTowerOfTwo(state1, state2)  or
        valid_FromOneTowerOfTwoToOneTowerOfThree(state1, state2) or
        valid_FromOneTowerOfThree(state1, state2)):

        return True

    return False

# print(check_if_valid_transition_BIS([0,1,2], [0,1,2]))
# print(check_if_valid_transition_BIS([0,1,2], [0,2,1]))



# def check_if_valid_transition(state1, state2):
#     nber_cleared_blocksState1 = len(state1)
#     nber_cleared_blocksState2 = len(state2)
#     if abs(nber_cleared_blocksState1 - nber_cleared_blocksState2) > 1:
#         return False
#     liste_clears_prev, liste_not_clears_prev = return_cleared_and_not_cleared_blocks(state1)
#     liste_clears_next, liste_not_clears_next = return_cleared_and_not_cleared_blocks(state2)
#     # check cleared BLOCKS in next state
#     for b in liste_clears_next:
#         if b in liste_not_clears_prev: # if b was not clear before
#             # block that was on top of b, SHOULD always be clear (before and now)
#             if not (block_on_top(b, state1) in liste_clears_prev and block_on_top(b, state1) in liste_clears_next):
#                 return False
#     # check UNcleared BLOCKS in next state
#     for b in liste_not_clears_next:
#         if b in liste_clears_prev:
#             # block that IS on top of b, should always be clear (before and now)
#             if not (block_on_top(b, state2) in liste_clears_prev and block_on_top(b, state2) in liste_clears_next):
#                 return False
#     return True
   

# [[0, 1, 2], [0, 2, 1]]
# 
#      [[0, 1, 2], ([0, 1], 2)]
#
#           de 3 à 3: UN SEUL doit avoir même indice
#
#           de 3 à 2: (donc de 3 à une tour de 2), 

all_combis = []

for state1 in all_states:

    for state2 in all_states:

        if state1 != state2:
            #print(state1) # [0, 1, 2]
            #print(state2)

            #if len(state1) ==  2 and len(state2) == 1:
            if check_if_valid_transition_BIS(state1, state2):
                if type(state1) is tuple: state1 = list(state1)
                if type(state2) is tuple: state2 = list(state2)
                all_combis.append([state1, state2])
            # else:
            #     print([state1, state2])


print("len all_combis") # 132 for 24 states (3 blocks)
print(len(all_combis))
print("all_combis")
print(all_combis[:15]) # (1, [0, 2])     [1, 2, 0]

# 1) no tower ===> all combis
# 2) one tower of size two : for each tower, take the rest and all combis
# 3) one tower of size three: IDEM
# 4) two towers


def load_blocks(track, num_examples, parameters=None, objects=True,**kwargs):
    #with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
    # dataset that produces images: ['images', 'bboxes', 'picsize', 'transitions']
    # my own dataset : ,['images_mean', 'images_var', 'coords_mean', 'coords_var', 'picsize', 'patch_shape', 'num_samples_per_state', 'lol', 'transitions', 'all_infos']
    #with np.load("/workspace/latplanDatasetGen/cylinders-2-flat.npz", allow_pickle=True) as data:
    
    with np.load("/workspace/latplanDatasetGen/20000-flat.npz", allow_pickle=True) as data:
        # 

        #print(data.keys())

        # Display the keys
        keys = data.files
        print(keys)
        # print(data["transitions"])


        images = data['images'].astype(np.float32) / 255

        print(images.shape) # (4, 1, 100, 150, 3) PAS BON

        all_relations = data["all_relations"]

        #print(data["all_descs"])
        all_descs = data["all_descs"]
        print("clears:")
        print(data["all_descs"][0]["clears"])
        print()
        print("xAboveY:")
        print(data["all_descs"][0]["xAboveY"])
        print()
        print("behind:")
        print(data["all_relations"][0]["behind"])
        print()
        print("front:")
        print(data["all_relations"][0]["front"])
        print()
        print("left:")        
        print(data["all_relations"][0]["left"])
        all_lefts = data["all_relations"]
        # [[1, 2], [], [1]] means on the left of 0 there is 1 and 2
        # On the left of 2 there is 1
        print()
        print("right:")
        print(data["all_relations"][0]["right"])


        # bboxes               = data['bboxes']
        all_transitions_idx  = data['transitions']
        print(len(all_transitions_idx))
      
        # picsize              = data['picsize']
        # num_states, num_objs = bboxes.shape[0:2]
        # print("loaded. picsize:",picsize)

    #parameters["picsize"] = [picsize.tolist()]
    #parameters["generator"] = None

    print(all_transitions_idx[:3]) # [0 1 2]

    all_transitions_idx = all_transitions_idx.reshape((len(all_transitions_idx)//2, 2))

    
    #np.random.shuffle(all_transitions_idx)
    transitions_idx = all_transitions_idx[:num_examples]

    # transitions_idx[i] is like [5,6]
    print(transitions_idx[0][0])
    print(transitions_idx[0][1])

    print("ici")
   
    print(images[transitions_idx[0][0]])

    plot_image(np.array(images[transitions_idx[0][0]]).squeeze(), "blocksPRE")
    plot_image(np.array(images[transitions_idx[0][1]]).squeeze(), "blocksSUCC")


    ##### ACTIONS

    # PREs clears
    # print(all_transitions_idx[:, 0])
    # print(all_descs[:5])
    print("hello")
   
    print("all_relations")

    left_elements = [d['left'] for d in all_relations]
    
    xAboveY_elements = [d['xAboveY'] for d in all_descs]

    # print(left_elements[16195])
    # print(xAboveY_elements[16195])
    # # [[1, 2], [], [1]]
    # # [[0, 2]]

    # print(images.shape) # (54324, 1, 30, 45, 3)

    # plot_image(np.squeeze(images[16195]),"AAAAAAAA")


    # (1, [0, 2])     [1, 2, 0]
    #fromTowerAndLeftsBuildState(tower, lefts)
    
    # below, does exactly what's supposed to do, i.e. return for a given xAboveY array, the corres tower
    alltowerss = return_all_towers_and_xAboveY()
    

    all_the_towers = []


    for ij, xAboveYs in enumerate(xAboveY_elements): # for each transition, we take the xAboveYs value

        # print("xAboveYs")
        # print(xAboveYs)

        did_find_tower=False


        if xAboveYs: # if the xAboveYs value is not an empty list (i.e. there is at least a block on top of another)

            for k, v in alltowerss.items(): # we loop over all the configurations of tower (e.g. { (1,0,2) : [1,0,2]}   )
                xAboveYs_is_v = True # we assume that the xAboveYs IS a tower (then we filter out)
                
                for el in xAboveYs: # for each element in xAboveYs (like [0, 1])
                    if el not in v: # if the el is not among elements in "v" , i.e. in the elements of the current configuration
                        
                        xAboveYs_is_v = False # THEN xAboveYs_is_v is False
                        break
                        #all_the_towers.append([])
                if xAboveYs_is_v: # here, it means, that all the elements in xAboveYs ARE ALSO in v
                    ss = k.strip("()") # From the key we construct the tower
                    numbersss = ss.split(", ")
                    resultsss = [int(num) for num in numbersss]
                    all_the_towers.append(resultsss)
                    did_find_tower=True
                    break

        if not did_find_tower or not xAboveYs:
            all_the_towers.append([])
       

    # 16195    ça fait une tour like [0, 2] MAIS left est [[1, 2], [], [1]] donc problem
    # 
    #towers = []

    # NOW... ALL THE STATES
    all_the_states = []
    print(len(all_the_towers)) # 54324

    # an array to store states that are not "logic"
    bad_indexes = []

    for iii in range(len(all_the_towers)):
        #print("iii : {}".format(str(iii)))
        thestate = fromTowerAndLeftsBuildState(all_the_towers[iii], left_elements[iii], iii)
        # if iii == 10000:
        #     exit()
        if thestate:
            all_the_states.append(thestate)
        else:
            all_the_states.append("BAD STATE")
            if iii%2 == 0:
                #bad_indexes.append([iii, iii+1])
                bad_indexes.append(iii)
                bad_indexes.append(iii+1)
            else:
                bad_indexes.append(iii)
                bad_indexes.append(iii-1)

    print(images.shape) # (54324, 1, 30, 45, 3)
    plot_image(np.squeeze(images[112]), "ABON112")
    print("all_the_states[112]")
    print(all_the_states[112])
    print()
    plot_image(np.squeeze(images[113]), "ABON113")
    print("all_the_states[113]")
    print(all_the_states[113])
    
    # !!!!!!!!!!!! all_the_states IS GOOD !!!!!!!!!!!!


    # si impair ALORS mettre son index MAIS aussi le -1
    # si pair 
    print("bad_indexes")
    print(len(bad_indexes))
    print("images ")

    # 

    images = rgb_to_grayscale(images)

    #plot_image(np.squeeze(images[16195]),"AAAAAAAA")

    images_cleaned = []
    for hhhhhh, im in enumerate(images):
        if hhhhhh not in bad_indexes:
            images_cleaned.append(im)

    all_the_states_cleaned = []
    for iu, sss in enumerate(all_the_states):
        if iu not in bad_indexes:
            all_the_states_cleaned.append(sss)
    

    all_unique_actions = []

    all_actions_indices = []

    print("commence")
    print(all_combis)
    print()
    all_actions = []
    for jjj in range(0, len(all_the_states_cleaned), 2):
        to_add = [all_the_states_cleaned[jjj], all_the_states_cleaned[jjj+1]]
        if to_add in all_combis:
            all_actions.append(to_add)
            all_actions_indices.append(all_combis.index(to_add))
        else:
            print("PROOOOBLEME")
        if to_add not in all_unique_actions:
            all_unique_actions.append(to_add)

    print("len all_unique_actions")
    print(len(all_unique_actions))
    print(all_actions_indices[:2])
    #exit()
    print()
    print(all_actions[2]) # [[2, [1, 0]], [[2, 1, 0]]]

    # 0 => 0 1
    # 1 => 2 3
    # 2 => 4 5
    plot_image(np.squeeze(images_cleaned[4]), "A4")
    plot_image(np.squeeze(images_cleaned[5]), "A5")
    

    images_squeezed = np.squeeze(images_cleaned)

    print(transitions_idx[:,0])
    print()
    print(transitions_idx[:,1])

    if objects:
        all_states = np.concatenate((images.reshape((num_states, num_objs, -1)),
                                     bboxes.reshape((num_states, num_objs, -1))),
                                    axis = -1)
        pres = all_states[transitions_idx[:,0]]
        sucs = all_states[transitions_idx[:,1]]
        transitions, states = normalize_transitions_objects(pres,sucs,**kwargs)
    else:
        pres = images_squeezed[transitions_idx[:,0]]
        sucs = images_squeezed[transitions_idx[:,1]]
        transitions, states, mean, std = normalize_transitions_here(pres, sucs)

    print("statess ????")
    #print(states[0])
    print("---------")
    # ACTION ???  
    print(transitions.shape) # (50, 2, 30, 45, 3)

    print(transitions[0][0])
    #exit()
    plot_image(transitions[22][0], "A22_PRE")
    plot_image(transitions[22][1], "A22_SUC")
    print("THE ACTION !!") # [[2, [1, 0]], [[2, 1, 0]]]

    # print(all_actions[22])
    # Combien d'actions ???
    # 132

    all_actions_onehot = F.one_hot(torch.tensor(all_actions_indices), num_classes=132)

    all_actions_onehot_numpy = []
    for a in all_actions_onehot:
        all_actions_onehot_numpy.append(a.numpy())

    all_actions_onehot_numpy = np.array(all_actions_onehot_numpy)
    #exit()7
    print("all_actions_onehot_numpy.shape")
    print(all_actions_onehot_numpy.shape)

    for hh in range(0, 1500, 500):
        print("for h = {}".format(str(hh)))
        #print(np.where(actions_one_hot.numpy()[hh] == 1)[0])
        print(all_combis[np.where(all_actions_onehot_numpy[hh] == 1)[0][0]])
    

    # Now transform the actions into a one hot encoding 

    return transitions[:num_examples], states[:num_examples], all_actions_onehot_numpy[:num_examples], mean, std


################################################################
# flat images

def return_blocks(args, parameters):
    parameters["generator"] = "latplan.puzzles.blocks"
    transitions, states, actions, mean, std = load_blocks(**vars(args), parameters=parameters, objects=False)

    return transitions, states, actions, mean, std
    #ae = run(os.path.join("samples",common.sae_path), transitions)



################################################################
# object-based representation

def blocks_objs(args):
    parameters["generator"] = "latplan.puzzles.blocks"
    transitions, states = load_blocks(**vars(args))

    ae = run(os.path.join("samples",common.sae_path), transitions)

    transitions = transitions[:6]
    _,_,O,_ = transitions.shape
    print("plotting interpolation")
    for O2 in [3,4,5]:
        try:
            masked2 = random_object_masking(transitions,O2)
        except Exception as e:
            print(f"O2={O2}. Masking failed due to {e}, skip this iteration.")
            continue
        ae.reload_with_shape(masked2.shape[1:])
        plot_autoencoding_image(ae,masked2,f"interpolation-{O2}")
    print("plotting extrapolation")
   
    pass

