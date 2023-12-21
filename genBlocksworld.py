import numpy as np
import os.path
import itertools
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


def plot_image(a,name):
    #plt.figure(figsize=(6,6))
    plt.figure()
    #plt.imshow(a,interpolation='nearest',cmap='gray',)
    plt.imshow(a)
    plt.savefig(name)


#### ALL POSSIBLE "CLEAR" combinations
def all_clear_combination():
    numbers = [0, 1, 2, 3]
    all_combinations = []
    for r in range(1, len(numbers) + 1):
        combinations = itertools.combinations(numbers, r)
        all_combinations.extend(combinations)
    return all_combinations


########################################

#### TOWERs combinations
def return_all_possible_towers():
    selected_permutations = []
    for r in range(2, 5):
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


################################################################################

# suffit de décrire les rights (les lefts sont le mirroir)

#       2 blocks qui sont en colonne ne peuvent être à right l'un autre

#        condition sur les rights : si a right of b alors b cannot right of a

#                                           a right of b, c right of 




# exceptions can be like [[1, 3, 2]]
def item_has_nothing_on_left_except(lefts_list, number_or_tower, exceptions=None):
    retour = True
    for i, l in enumerate(lefts_list):
        if type(number_or_tower) is list and len(number_or_tower) > 0:
            reference_block = number_or_tower[0]
        else:
            reference_block = number_or_tower
        if exceptions==None:
            if i == reference_block and l:
                retour = False
        else:
            if i == reference_block and l:
                eles_not_in_exception = True
                flatten_exceptions = flatten(exceptions)
                for el in l:
                    if el not in flatten_exceptions:
                        retour = False
    return retour

# all combinations of towers with 2 to 4 numbers
#     then for each, retrieve the remaining

# input like tower=[2, 0] and  lefts=[[1,3],[3],[1,3],[]]
# (represents 3 1 2/0 )
# output is [3, 1, [2, 0]]
def fromTowerAndLeftsBuildState(tower, lefts):
    import numpy as np
    final_state = []
    if type(tower) is list:
        if tower and len(tower) ==  2:
            if type(tower[0]) is list and type(tower[1]) is list:
                
                    # just determine which tower is on left of the other
                    # take 1st element of tower 1 and see if has smth on its left
                    #tower[0][0]
                    if lefts[tower[0][0]]:
                        final_state.append(tower[1])
                        final_state.append(tower[0])
                    else:
                        final_state.append(tower[0])
                        final_state.append(tower[1])
                    return final_state

    # 
    remaining_elements_of_state = []
    for i in range(4):
        if i not in tower:
            remaining_elements_of_state.append(i)
    if tower:     
        remaining_elements_of_state.append(tower)

    present_elements = remaining_elements_of_state.copy()
    placed_elements = []
    counter=0
    # print("remaining_elements_of_state")
    # print(remaining_elements_of_state)
    while(len(final_state) != len(present_elements)):
        #print("counter : {}".format(str(counter)))
        #print(remaining_elements_of_state)
        for el in remaining_elements_of_state:
            if el not in final_state and item_has_nothing_on_left_except(lefts, el, exceptions=placed_elements):
                final_state.append(el)
                placed_elements.append(el)
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
            #if len(state1) ==  2 and len(state2) == 1:
            if check_if_valid_transition_BIS(state1, state2):
                all_combis.append([state1, state2])
            # else:
            #     print([state1, state2])




# 1) no tower ===> all combis
# 2) one tower of size two : for each tower, take the rest and all combis
# 3) one tower of size three: IDEM
# 4) two towers


# for each state, 

# the ids are 0 1 2 3
#
#    e.g. clears = [1, 2]
#      xAboveY = [[1, 0], [....]] # à vérifier !!!   6 combinaisons max

#     lefts = [[]]
#                       [[1, 2], [], [1]]
#                                           for each combination
#                                                                  e.g. 1 2 0
#                                                                       for each item
#                                                                           
#      IDEM pour rights et behind et front


# construire tout les scénarios possibles, 
#  où un scénario is like [1, 0, [2, 3]], i.e. represent vraiment une config = FACILE
#
#
#                   puis à partir de chaque exemple du dataset, 

#
#                               1) construire la tour avec les xAboveY
#                               2) construire le reste avec les lefts
#
#                                           i) si tour existe, prend un ele de la tour, et place les éléments qui sont à gauche
#                                           ii) parmis les élé qui sont à gauche, place les + à gauche


def load_blocks(track, num_examples, parameters=None, objects=True,**kwargs):
    #with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
    # dataset that produces images: ['images', 'bboxes', 'picsize', 'transitions']
    # my own dataset : ,['images_mean', 'images_var', 'coords_mean', 'coords_var', 'picsize', 'patch_shape', 'num_samples_per_state', 'lol', 'transitions', 'all_infos']
    #with np.load("/workspace/latplanDatasetGen/cylinders-2-flat.npz", allow_pickle=True) as data:
    with np.load("/workspace/latplanDatasetGen/cylinders-3-flat.npz", allow_pickle=True) as data:
        # 

        #print(data.keys())

        # Display the keys
        keys = data.files
        print(keys)
        # print(data["transitions"])

        #print(data["lol"])
        # immmms = data['images'].astype(np.float32) / 255
        # print(type(immmms)) # ORIGINAL IMAGE DATA nparray
        # print(immmms.shape) # ORIGINAL IMAGE DATA  (80000, 1, 30, 45, 3)

        # # Don't forget to close the file after you're done
        # print(data["all_descs"])
        # data.close()
        # #

        images               = data['images'].astype(np.float32) / 255
        print("okk")
        print(images.shape) # (4, 1, 100, 150, 3) PAS BON

        all_relations = data["all_relations"]

        #print(data["all_descs"])
        all_descs=data["all_descs"]
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
        # [[1, 2], [], [1]] means on the left of 0 there is 1 and 2
        # On the left of 2 there is 1
        print()
        print("right:")
        print(data["all_relations"][0]["right"])
        # [[], [0, 2], [0]] means on the right of 1 there is 0 and 2
        #                       on the right of 2 there is 0    

        # clears = data["all_descs"]["clears"]
        # xabovey = data["all_descs"]["xAboveY"]

        # bboxes               = data['bboxes']
        all_transitions_idx  = data['transitions']
        # picsize              = data['picsize']
        # num_states, num_objs = bboxes.shape[0:2]
        # print("loaded. picsize:",picsize)

    # print("tyyypes")
    # print(type(relationships))
    # print(type(clears))
    # print(type(xabovey))


    #parameters["picsize"] = [picsize.tolist()]
    #parameters["generator"] = None

    all_transitions_idx = all_transitions_idx.reshape((len(all_transitions_idx)//2, 2))

    # all_transitions_idx[i] return the two indices of the images of the pair i
    print("all_transitions_idx")
    print(all_transitions_idx[:5])

    np.random.shuffle(all_transitions_idx)
    transitions_idx = all_transitions_idx[:num_examples]

    print(transitions_idx)

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
    print(all_descs[all_transitions_idx[:, 0]])

    # 


    print(all_pre_clears[:5])

    if objects:
        all_states = np.concatenate((images.reshape((num_states, num_objs, -1)),
                                     bboxes.reshape((num_states, num_objs, -1))),
                                    axis = -1)
        pres = all_states[transitions_idx[:,0]]
        sucs = all_states[transitions_idx[:,1]]
        transitions, states = normalize_transitions_objects(pres,sucs,**kwargs)
    else:
        pres = images[transitions_idx[:,0],0]
        sucs = images[transitions_idx[:,1],0]
        transitions, states = normalize_transitions(pres,sucs)

    print("statess ????")
    print(states[0])
    print("---------")
    exit()

    return transitions, states


################################################################
# flat images

def return_blocks(args, parameters):
    parameters["generator"] = "latplan.puzzles.blocks"
    transitions, states = load_blocks(**vars(args), parameters=parameters, objects=False)

    return transitions
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

