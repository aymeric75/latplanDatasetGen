import numpy as np
import os.path
import torch
import sys
#sys.path.append(r"..latplanClonedLossTerm/latplan")
sys.path.insert(0, '/workspace/latplanClonedEnforce')
import latplan
from latplan.puzzles.objutil import image_to_tiled_objects, tiled_bboxes
from latplan.main.normalization import normalize_transitions, normalize_transitions_objects
sys.path.remove('/workspace/latplanClonedEnforce')

import matplotlib.pyplot as plt

nb_towers=4
nb_rings=4



###################### UTIL FUNCTIONS ######################


# Convert to grayscale using the luminosity method
def rgb_to_grayscale(rgb_images):
    grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07])
    return np.stack((grayscale,)*3, axis=-1)



def return_flatttt(s1, s2):
    retour=[]
    for ju in s1:
        retour.append(ju)
    for jk in s2:
        retour.append(jk)
    return np.array(retour)



def plot_image(a,name):
    #plt.figure(figsize=(6,6))
    plt.figure()
    #plt.imshow(a,interpolation='nearest',cmap='gray',)
    plt.imshow(a, interpolation='nearest')
    plt.savefig(name)

def plot_one_line(a, name):

    plt.plot(a[0])
    plt.show()
    plt.subplot(1,2,1)
    plt.plot(a[1])
    plt.subplot(1,2,2)
    plt.plot(a[2])
    plt.show()

import torch.nn.functional as F



def normalize(x,save=True):

    mean               = np.mean(x,axis=0)
    std                = np.std(x,axis=0)
    # if save:
    #     parameters["mean"] = [mean.tolist()]
    #     parameters["std"]  = [std.tolist()]
    print("normalized shape:",mean.shape,std.shape)
    #return (x - mean)/(std+1e-20)
    return mean, std

def return_mean_and_var(pres, sucs):

    transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
    B, *F = pres.shape
    mean, std  = normalize(np.reshape(transitions, [-1, *F])) # [2BO, F]
    return mean, std


##############################################################


############### ToH PDDL FUNCTIONS ####################
# 
# state from the dataset given as [d1, d2, d3, d4], 
# e.g. [ 1 1 0 3 ] disk0 at tower1, disk1 at tower1, disk2 at tower 0, disk3 at tower 3...
# where disk0 is the smallest (white), disk3 is the largest (blue)


# for 4 disks
def generate_all_possible_states_hanoi(nb_towers):
    all_configs = []
    # disk1 position
    for d1 in range(nb_towers):
        # disk2 position
        for d2 in range(nb_towers):
            # disk3 position
            for d3 in range(nb_towers):
                # disk3 position
                for d4 in range(nb_towers):
                    oneconfig = [d1, d2, d3, d4] # [ 1, 1, 1, 1 ]
                    all_configs.append(oneconfig)
    return all_configs


# stuff is either a disk either a tower
def clear(disc, state):

    if type(disc) is list:
        if disc[0] in state:
            return False
        return True

    tower_index = state[disc]
    discs_at_tower_name = [i for i, st in enumerate(state) if st == tower_index]
    for d in discs_at_tower_name:
        if d < disc:
            return False
    return True


def top_of_tower(tower_number, state):

    discs_at_tower_number = [i for i, st in enumerate(state) if st == tower_number]
    if discs_at_tower_number:
        top_disc = discs_at_tower_number[0]
    else:
        # if tower is empty, we return a list with the tower_number in it
        top_disc = [tower_number]

    return top_disc



# [0,1,2,3]  [0,1,2,3]

def return_moving_disc(state1, state2):
    moving_discs = []
    counter = 0
    for i, s1 in enumerate(state1):
        if s1 != state2[i]:
            counter+=1
            moving_discs.append(i)
    if counter == 0:
        return False
    if counter > 1:
        return False
    if counter == 1:
        return moving_discs[0]
   

# [1,1,2,3], [0,1,2,3]
def return_pre_and_suc_tour(state1, state2):
    for i, s1 in enumerate(state1):
        if s1 != state2[i]:
            return s1, state2[i]
    return None

# return list representing the disk (from smaller to bigger)
# presents in the tour (tour_number), at state "state"
def return_tour_desc(tour_number, state):
    desc = []
    for i, s in enumerate(state):
        if s == tour_number:
            desc.append(i)
    return desc

def return_stack_description_of_pre_and_suc_tour(state1, state2):
    tour_pre_desc = []
    tour_suc_desc = []
    for i, s1 in enumerate(state1):
        if s1 != state2[i]:
            tour_pre, tour_suc = s1, state2[i]

            tour_pre_desc = return_tour_desc(tour_pre, state1)

            tour_suc_desc = return_tour_desc(tour_suc, state1)

    return tour_pre_desc, tour_suc_desc


def smaller(ele1, ele2):
    if type(ele1) is list and type(ele2) is int:
        return True
    if ele2 < ele1:
        return True
    return False



# say if disc1 is on disc2
# 
def on(disc1, disc2, state):
    #top_ele = top_of_tower(tower_index, state)

    if type(disc1) is list:
        return False

    if type(disc2) is list:
        # 
        tower_nb = disc2[0]
        if state[disc1] == tower_nb:
            return True
        return False

    if state[disc1] == state[disc2] and disc1 < disc2:
        return True
    return False


def return_element_below_top(from_tour, state):
    top = top_of_tower(from_tour, state)
    discs_at_tower_number = [i for i, st in enumerate(state) if st == from_tour]
    
    if len(discs_at_tower_number) == 0:
        return [from_tour]
    else:
        discs_at_tower_number.remove(top)
        if len(discs_at_tower_number) > 0:
            return discs_at_tower_number[0]
        else:
            return [from_tour]
    

def is_valid_move(state1, state2):

    # id le num du disc qui bouge
    disc = return_moving_disc(state1, state2)
    

    if type(disc) is bool and disc is False:
        return False, None, None

    # id les num des deux tours
    # from1, to should be the top of the two tours
    from_tour, to_tour = return_pre_and_suc_tour(state1, state2)
    #print("from_tour: {}, to_tour: {}".format(str(from_tour), str(to_tour)))
    from1 = return_element_below_top(from_tour, state1)
    to = top_of_tower(to_tour, state1)

    pre = (smaller(to, disc) and on(disc, from1, state1) and
        clear(disc, state1) and clear(to, state1)    
    )

    effect = (
        clear(from1, state2) and on(disc, to, state2) and 
        (not on(disc, from1, state2)) and
        (not clear(to, state2))
    )
  
    if pre and effect:
        # pre_state = [m=> name_of_tour !!!!!!!!!!!!!
        # fully specify the tower of start and of end  !!!!!!!
        # 
        # Now, 
        tour_pre_desc, tour_suc_desc = return_stack_description_of_pre_and_suc_tour(state1, state2)
        
        # e.g. "move DISC from TOUR_fully_Specified at to TOUR_fully_specified at "
        middle_specified_action = "move_"+str(disc)+"_from_"+str(tour_pre_desc)+"_at_"+str(from_tour) + "_to_" +str(tour_suc_desc)+"_at_"+ str(to_tour)

        full_preState_desc = str(state1)+"_"+"move_"+str(disc)+"_from_"+str(from_tour) + "_to_" + str(to_tour)

        return True, "move_"+str(disc)+"_from_"+str(from_tour) + "_to_" + str(to_tour), middle_specified_action

    return False, None, None



################ STATE AND ACTION GENERATION ################


all_hanoi_states = generate_all_possible_states_hanoi(nb_towers)


all_actions_names = []
all_hanoi_transitions_pddl_constraints = []


### All states generation
for s1 in all_hanoi_states:

    for s2 in all_hanoi_states:
        
        if s1 != s2:
            # name_middle_specified_action
            valid, name_loose, name_middle_specified_action = is_valid_move(s1, s2)
            if valid:
        
                all_hanoi_transitions_pddl_constraints.append(str(return_flatttt(s1, s2)))
                if name_middle_specified_action not in all_actions_names:
                    all_actions_names.append(name_middle_specified_action)




######################## THE FUNCTION THAT RETURN EVERYTING ########################
def return_hanoi_transitions_and_actions(args, parameters, version="simple", masks=True):
    
    parameters["generator"] = "latplan.puzzles.hanoi"
    disks  = args.disks
    towers = args.towers
    num_examples = args.num_examples
    num_examples = 20000
    import latplan.puzzles.hanoi as p
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]


    ## CALCULATE FOR EACH PAIR THE FROM and TOs (retrieved_items_*)
    diff_between_arrays = pre_configs - suc_configs # still a numpy array
    # in order to identify what ring will move (e.g. [0 2 0 0], ie disk number 1 (red) moved)
    diff_between_arrays_abs = np.absolute(diff_between_arrays)
    # index where not 0, i.e. index of the disk (i.e. its color!) that was moved
    # e.g [1 0 3 2 ....]
    non_zero_indices = np.argmax(diff_between_arrays_abs, axis=1)
    # = for each pair, the position "FROM" (i.e. the index of the tower where the disk was picked)
    retrieved_items_pre_configs = [pre_configs[i, non_zero_indices[i]] for i in range(len(pre_configs))]
    # = for each pair, the position "TO"
    retrieved_items_suc_configs = [suc_configs[i, non_zero_indices[i]] for i in range(len(suc_configs))]


    #### TO REMOVE ? (not used in the rest of the code)
    # at each "line" contains the list of disk present on the pre tower
    # all_pre_disks = []
    # for i, e in enumerate(pre_configs):
    #     # Find indices where the elements in pre_configs[i] are equal to e
    #     indices_equal_to_e = np.where(e == retrieved_items_pre_configs[i])[0]
    #     thestring = ''.join([str(ind) for ind in sorted(indices_equal_to_e, reverse=True)])
    #     thestring = int(thestring)
    #     all_pre_disks.append(thestring)
    # # at each "line" contains the list of disk present on the pre tower
    # all_succ_disks = []
    # for i, e in enumerate(pre_configs):
    #     # Find indices where the elements in pre_configs[i] are equal to e
    #     indices_equal_to_e = np.where(e == retrieved_items_suc_configs[i])[0]
    #     if indices_equal_to_e.any():
    #         thestring = ''.join([str(ind) for ind in sorted(indices_equal_to_e, reverse=True)])
    #         thestring = int(thestring)
    #     else:
    #         thestring = 9
    #     all_succ_disks.append(thestring)
    

    ### BUILDING ARRAYS OF ALL ACTIONS (from all transitions) wth diff lvl of DESCRIPTIONS

    all_actions_full_desc = np.column_stack((pre_configs, suc_configs))
    all_actions_pddl_constraints = np.column_stack((pre_configs, suc_configs))
    all_actions_full_preState_desc_pddl_desc = []
    all_actions_loose_pddl_desc = []

    for ij in range(len(pre_configs)):
        ___, name_loose, name_middle_specified_action = is_valid_move(pre_configs[ij], suc_configs[ij])
        #all_actions_full_preState_desc_pddl_desc.append(name_full_preState)
        all_actions_loose_pddl_desc.append(name_middle_specified_action)
    
    all_simple_actions = np.column_stack((retrieved_items_pre_configs, retrieved_items_suc_configs))


    # TO REMOVE ????? 
    #rings_on_target = [np.where(row == value)[0] for row, value in zip(pre_configs, retrieved_items_suc_configs)]
    # topring_on_target_ = []
    # sorted_rings_on_target_ = []
    # for li in rings_on_target:
    #     if li.size == 0:
    #         topring_on_target_.append(9)
    #         sorted_rings_on_target_.append(9)
    #     else:
    #         li = sorted(li, reverse=True)
    #         li_tmp = [str(l) for l in li]
    #         topring_on_target_.append(np.min(li))
    #         stacked_target = int(''.join(li_tmp))
    #         if stacked_target == 320:
    #             print("stacked target 320 detected")
    #         sorted_rings_on_target_.append(stacked_target)

    # print("LOLILOL")
    # print(all_actions_loose_pddl_desc[:3])
    # print("AAAAAAAABBBBBBBBBBBBBBBBBBBBBB")
    # print(all_actions_names[:3])
    # exit()
    ####### ASSIGNING all the actions & combinations to all_actions and all_combis
    if version == "simple":
        all_actions = all_actions_loose_pddl_desc
        all_combis = all_actions_names

    elif version == "with_loose_pddl_desc":
        all_actions = all_actions_loose_pddl_desc
        #flattened = [np.array(sub_array).flatten() for sub_array in all_hanoi_transitions_pddl_constraints]
        all_combis = all_actions_names

    elif version == "with_preStateFull_pddl_desc":
        all_actions = all_actions_full_preState_desc_pddl_desc
        all_combis = all_actions_names


    ##### TURNING ACTIONS INTO ONE-HOT

    # hold, for each transition pair, the index of the action from the
    # actions' label vector, i.e. from all_combis_simple or ...
    indices = []

    ###### JUST a test to count the occurence of each action in the dataset
    dicc = {}
    for ac in all_actions:
        if str(ac) in dicc.keys():
            dicc[str(ac)] += 1
        else:
            dicc[str(ac)] = 1
        


    # in all_actions at each line, there is a desc of an action (e.g. [3, 0])
    # we accumulate the index of the action (the latter being the one of all_combis)
    for ccc, row in enumerate(all_actions):
        #print("ccc : {}".format(str(ccc)))
        smth_was_found=False
        if str(row) in all_combis:
            indices.append(all_combis.index(str(row)))
            smth_was_found=True
        if not smth_was_found:
            print("therow of element {}".format(str(ccc)))
            print(row)
            print("all_combis")
            print(all_combis)
            exit()

    # contains, for all transition of the dataset, the index of the <=> action (the index in all_combis)
    actions_indexes = torch.tensor(indices)
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_combis))

    # ####### ANOTTHER TEST: to check if the sum of one hots equals the total number of combis    
    # actions_summed = torch.sum(actions_one_hot, dim=0)
    # nber_of_actions_sampled = torch.sum(actions_summed)
    # print(actions_summed)
    # print(len(actions_summed))
    # print(nber_of_actions_sampled) # 5000

    ####### GENERATING THE IMAGES
    pres = p.generate(pre_configs, disks, towers)
    sucs = p.generate(suc_configs, disks, towers)

    print("prespresprespres")
    print(pres[0])

    # rgb_to_grayscale
    pres = rgb_to_grayscale(pres)
    sucs = rgb_to_grayscale(sucs)
    print(pres.shape)
    #exit()
    parameters["picsize"] = pres[0].shape
    print("PUTAIN0")
    mean_to_use, std_to_use = None, None
    transitions, states, mean_to_use, std_to_use = normalize_transitions(pres, sucs)
    print("PUTAIN1")
    for hh in range(0, 1500, 500):
        print("for h = {}".format(str(hh)))
        #print(np.where(actions_one_hot.numpy()[hh] == 1)[0])
        print(all_combis[np.where(actions_one_hot.numpy()[hh] == 1)[0][0]])
    

    if masks:
        
        diff_transitions = np.abs(transitions[:, 0, :, :, :] - transitions[:, 1, :, :, :])
        modified_array = np.where(diff_transitions != 0, 1, diff_transitions)
        return transitions, actions_one_hot.numpy(), modified_array, mean_to_use, std_to_use

    else:

        return transitions, actions_one_hot.numpy(), mean_to_use, std_to_use

