import numpy as np
import os.path
#from ..util.tuning import parameters
#from .common import *
#from .normalization import normalize_transitions
#from . import common
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


all_combis_simple = []

for i in range(nb_towers):
    for j in range(nb_towers):
        if i != j:
            all_combis_simple.append([i, j])

# 





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
                    oneconfig = [d1, d2, d3, d4]
                    all_configs.append(oneconfig)
    return all_configs

def check_if_disk_above(conf_dep, conf_dest):
    index = np.where(np.abs(np.array(conf_dep) - np.array(conf_dest)) != 0)[0][0]
    # ex conf = [0, 0, 0, 0], index = 3
    has_smth_above = False
    for sub_index in range(index):
        if conf_dep[sub_index] == conf_dep[index]:
            has_smth_above = True
    return has_smth_above


def check_destination(conf_dep, conf_dest):
    index = np.where(np.abs(np.array(conf_dep) - np.array(conf_dest)) != 0)[0]
    if len(list(index)) == 1:
        #print(index)
        dest_tower  = conf_dest[index[0]] # 1
    else:
        return False
    indices = np.where(np.array(conf_dep) == dest_tower)
    destination_safe = True
    if len(indices) > 0:
        for ind in indices[0]:
            if ind < index:
                destination_safe = False
    return destination_safe


def generate_all_transitions(nb_towers):
    all_states = generate_all_possible_states_hanoi(nb_towers)
    all_transitions = []
    for pre in all_states:
        for suc in all_states:
            if check_destination(pre, suc) and not check_if_disk_above(pre, suc):
                #all_transitions.append([pre, suc])
                the_trans = []
                for p in pre:
                    the_trans.append(p)
                for s in suc:
                    the_trans.append(s)
                all_transitions.append(np.array(the_trans))
                #all_transitions.append(np.array([pre, suc]))
    return all_transitions
    




# from a given left stack, builds all the combinations for
# the right stack
def build_right_stacks_from_left(left_stack):
    # 310
    can_use = [0,1,2,3]

    top_left_stack = int(left_stack[-1])
    
    # remove eles that are already in left stack
    for i in range(len(left_stack)):
        try:
            can_use.remove(int(left_stack[i]))
        except:
            a=1
    
    # only keep eles that are bigger than top of left stack
    can_use_last = []
    for el in can_use:
        if top_left_stack < el:
            can_use_last.append(el)

    # # 
    right_stacks = []
    can_use_last = sorted(can_use_last, reverse=True)
    # # [3, 2, 1]
    for i in range(len(can_use_last)):

        for j in range(i, len(can_use_last)):
            if can_use_last[i] != can_use_last[j]:
                right_stacks.append(can_use_last[i]*10+can_use_last[j])

    #print(right_stacks)

    for e in can_use_last:
        right_stacks.append(e)

    if can_use_last:
        biggest = int(''.join([str(i) for i in can_use_last]))
        if biggest not in right_stacks:
            right_stacks.append(int(''.join([str(i) for i in can_use_last])))

    # [3, 1]
    # 31, 3, 1

    return right_stacks



def build_left_stacks(ele_to_move):

    left_stacks=[]

    for i in range(ele_to_move+1, 4):
        #acc_rest.append(i)
        left_stacks.append(i)

    
    big_stacks = []
    if len(left_stacks) > 1:
        left_stacks = sorted(left_stacks, reverse=True)
        #print(left_stacks)
        for ii in range(len(left_stacks)):
            for jj in range(ii+1, len(left_stacks)):
                #print("ii: {}, jj:{}".format(str(ii), str(jj)))
                if left_stacks[ii] > left_stacks[jj]:
                    big_stacks.append(left_stacks[ii]*10 + left_stacks[jj])

        strs = [str(i) for i in left_stacks]
        #print()
        longest = int(''.join(strs))
        if longest not in big_stacks:
            big_stacks.append(int(''.join(strs)))

    #print(big_stacks)

    left_stacks_final=[]

    for el in big_stacks:
        
        left_stacks_final.append(el*10 + ele_to_move)

    left_stacks_final.append(ele_to_move)

    for e in left_stacks:
        last = e*10 + ele_to_move
        if last not in left_stacks_final:
            left_stacks_final.append(last)
    return left_stacks_final



print("all possible combis")
print(len(all_combis_simple))
print()
print(all_combis_simple)

# first ele : circle to pick up
# last ele: circle target or no circle (9)
# all_combis_with_prering_and_sucring

all_combis_with_prering_and_sucring = []

for i in range(nb_towers):
    for j in range(nb_towers):
        if i != j:
            for k in range(nb_rings):
                for kk in range(nb_rings+1):
                    if kk < nb_rings:
                        # blue(3) green(2) red(1) white (0): du plus gros au plus petit
                        if k < kk:
                            all_combis_with_prering_and_sucring.append([k, i, j, kk])
                            # identifier les nbres > à kk qui sont dans l'intervalle [0 - 3
                    else:
                        all_combis_with_prering_and_sucring.append([k, i, j, 9])






all_combis_with_prering_and_sucring_and_stack = []
# = also a description of the "to" stack

for i in range(nb_towers):
    for j in range(nb_towers):
        if i != j:
            for k in range(nb_rings):
                for kk in range(nb_rings+1):
                    if kk < nb_rings:
                        # blue(3) green(2) red(1) white (0): du plus gros au plus petit
                        if k < kk:
                            all_combis_with_prering_and_sucring_and_stack.append([k, i, j, kk])
                            # identifier les nbres > à kk qui sont dans l'intervalle [0 - 3]

                            if k==0 and kk==1:
                                all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 321])
                                all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 31])
                                all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 21])

                            if k==0 and kk==2:
                                all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 32])

                            if k==1 and kk==2:
                                all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 32])
                    else:
                        all_combis_with_prering_and_sucring_and_stack.append([k, i, j, 9])




print(all_combis_with_prering_and_sucring_and_stack)
print("lolilol")
print(len(all_combis_with_prering_and_sucring_and_stack)) # 192

print(all_combis_with_prering_and_sucring_and_stack[129])



all_combis_with_prering_and_sucring_and_both_stack = []
# = also a description of the "to" stack



for i in range(nb_towers):
    for j in range(nb_towers):
        if i != j:
            for k in range(nb_rings):
                #print("kkkk ")

                possible_left_stacks = build_left_stacks(k)

                #print(possible_left_stacks)

                for left_stack in possible_left_stacks:

                    possible_right_stacks = build_right_stacks_from_left(str(left_stack))

                    for right_stack in possible_right_stacks:

                        all_combis_with_prering_and_sucring_and_both_stack.append([left_stack, i, j, right_stack])
                    
                    all_combis_with_prering_and_sucring_and_both_stack.append([left_stack, i, j, 9])


print(all_combis_with_prering_and_sucring_and_both_stack)


all_combis_full_desc = generate_all_transitions(4)

### # blue(3) green(2) red(1) white (0): du plus gros au plus petit

#   [1, 2, 3, 32]
#       
#
#
#   [21, 2, 3, 3]
#       [21, 2, 3, 3] 
#
#           0123
#
#               1 to be moved
#
#                left: 1, 31, 21, 321 




#
#                       for each left
#
#                                   for instance 21: has to be with 0 and 3 at the max
#                                                                   (= the other numbers)
#
#                                               AND with bigger numbers: so 3 only and 9
#   [321, 2, 3, 32]

#   [31, 2, 3, ]


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

def return_hanoi_transitions_and_actions(args, parameters, version="simple"):
    
    parameters["generator"] = "latplan.puzzles.hanoi"
    disks  = args.disks
    towers = args.towers
    num_examples = args.num_examples
    num_examples = 10000
    import latplan.puzzles.hanoi as p
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    with np.load(path) as data:
       
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]

    print("ici")
    print(pre_configs[0])
    # 
    print()
    print(suc_configs[0])

    diff_between_arrays = pre_configs - suc_configs # still a numpy array
    print()
    # print("diff arrays")
    # print(diff_between_arrays[:5])

    # in order to identify what ring will move (e.g. [0 2 0 0], ie disk number 1 (red) moved)
    diff_between_arrays_abs = np.absolute(diff_between_arrays)

    # index where not 0, i.e. index of the disk (i.e. its color!) that was moved
    # e.g [1 0 3 2 ....]
    non_zero_indices = np.argmax(diff_between_arrays_abs, axis=1)
    print()
    # print("indices where change occured")
    # print(non_zero_indices[:5])
    
    # = for each pair, the position "FROM" (i.e. the index of the tower where the disk was picked)
    retrieved_items_pre_configs = [pre_configs[i, non_zero_indices[i]] for i in range(len(pre_configs))]
    
    # = for each pair, the position "TO"
    retrieved_items_suc_configs = [suc_configs[i, non_zero_indices[i]] for i in range(len(suc_configs))]

    print(pre_configs[17])
    print(suc_configs[17])
    print()
    #print(non_zero_indices[0])

    # from retrieved_items_pre_configs[i] (e.g. 3, i.e. the position of the "first" tower) look into pre_configs
    # np.where(retrieved_items_pre_configs == pre_configs, axis=1)
    
    # at each "line" contains the list of disk present on the pre tower
    all_pre_disks = []
    for i, e in enumerate(pre_configs):
        # Find indices where the elements in pre_configs[i] are equal to e
        indices_equal_to_e = np.where(e == retrieved_items_pre_configs[i])[0]
        thestring = ''.join([str(ind) for ind in sorted(indices_equal_to_e, reverse=True)])
        thestring = int(thestring)

        all_pre_disks.append(thestring)
    
    print("pre_disks")
    print(all_pre_disks[17])


    # at each "line" contains the list of disk present on the pre tower
    all_succ_disks = []
    for i, e in enumerate(pre_configs):
        # Find indices where the elements in pre_configs[i] are equal to e
        indices_equal_to_e = np.where(e == retrieved_items_suc_configs[i])[0]
        if indices_equal_to_e.any():
            thestring = ''.join([str(ind) for ind in sorted(indices_equal_to_e, reverse=True)])
            thestring = int(thestring)
        else:
            thestring = 9
        all_succ_disks.append(thestring)
    

    print("succ_disks")
    print(all_succ_disks[:20])



    ### => returns the indices from pre_configs that correspond to the disks of the FROM tower
    ### ==> mettre ces indices dans l'ordre descroissant et les "ajouter" (*10 + ...)
    ####### ==> PUIS même chose avec np.where(retrieved_items_suc_configs == pre_configs, axis=1)

    # print()
    # print("elements of pre that was moved")
    # print(retrieved_items_pre_configs[:5])
    # print()
    # print("elements of suc that was moved")
    # print(retrieved_items_suc_configs[:5])

    # like np.array([[3, 3, 2, 3], [0, 3, 2, 3], .....])
    all_actions_full_desc = np.column_stack((pre_configs, suc_configs))
    
    all_simple_actions = np.column_stack((retrieved_items_pre_configs, retrieved_items_suc_configs))
    print(all_simple_actions[:5])

    all_actions_with_prering = np.column_stack((non_zero_indices, all_simple_actions))
    print(all_actions_with_prering[:5])

    # reviewedActions-StackFrom-StackTo


    # determining the "value" (i.e. if ring or not and if yes which one) of the targetted tower
    # 1) retrieve the "tower" that is modified into the suc state ===> retrieved_items_suc_configs
    #  2) from the tower found in 1), look if this tower number present in the preconfig i) if not so 9 ii) if YES
    #### (if the "tower number" is presents multiple times, it means that I should fond which one on top)
    # which INDEX ? ===> gives the color (the last number to put on the right of the most complex action)
    
    # 1)
    #retrieved_items_suc_configs

    # 2) each item describes, for each trans, the list of rings already present in the target tower
    #### the one on top is the smallest value
    rings_on_target = [np.where(row == value)[0] for row, value in zip(pre_configs, retrieved_items_suc_configs)]

    topring_on_target_ = []
    sorted_rings_on_target_ = []
    for li in rings_on_target:
        if li.size == 0:
            topring_on_target_.append(9)
            sorted_rings_on_target_.append(9)
        else:
            # re order li desc
            # fusion as a string
            #     cast into int
            #li.sort(reverse=True)
            li = sorted(li, reverse=True)
            li_tmp = [str(l) for l in li]
            
            topring_on_target_.append(np.min(li))

            stacked_target = int(''.join(li_tmp))
            if stacked_target == 320:
                print("stacked target 320 detected")
            sorted_rings_on_target_.append(stacked_target)

    #pre_configs


    all_actions_with_prering_and_sucring = np.column_stack((all_actions_with_prering, topring_on_target_))

    all_actions_with_prering_and_sucring_and_stack = np.column_stack((all_actions_with_prering, sorted_rings_on_target_))

    print(all_actions_with_prering_and_sucring_and_stack[:5])


    all_actions_with_prering_and_sucring_and_both_stack = np.column_stack((all_pre_disks, all_simple_actions))
    all_actions_with_prering_and_sucring_and_both_stack = np.column_stack((all_actions_with_prering_and_sucring_and_both_stack, all_succ_disks))

    if version == "simple":
        all_actions = all_simple_actions
        all_combis = all_combis_simple
    elif version == "with_prering":
        all_actions = all_actions_with_prering
        all_combis = all_combis_with_prering
    elif version == "with_prering_and_sucring":
        all_actions = all_actions_with_prering_and_sucring
        all_combis = all_combis_with_prering_and_sucring
    elif version == "with_prering_and_sucring_and_stack":
        all_actions = all_actions_with_prering_and_sucring_and_stack
        all_combis = all_combis_with_prering_and_sucring_and_stack
    elif version == "with_prering_and_sucring_and_both_stack":
        all_actions = all_actions_with_prering_and_sucring_and_both_stack
        all_combis = all_combis_with_prering_and_sucring_and_both_stack
    elif version == "with_full_desc":
        all_actions = all_actions_full_desc
        all_combis = all_combis_full_desc



    # TURNING ACTIONS INTO ONE-HOT

    # hold, for each transition pair, the index of the action from the
    # actions' label vector, i.e. from all_combis_simple or ...
    indices = []

    print("all_actions")
    #print(all_actions)
    print(len(all_actions)) # 30000

    dicc = {}
    for ac in all_actions:
        if str(ac) in dicc.keys():
            dicc[str(ac)] += 1
        else:
            dicc[str(ac)] = 1
        
    # print("DICC")
    # print(dicc) # count the occurence of each action in the dataset
    # print(len(dicc)) # here 120 actions out of 192 samples .... that's a problem !!!
    ###################################  

    print("len(all_combis)")
    print(len(all_combis))

    print()
    print(all_actions[:2])

    print(len(all_actions)) # 10000

    # in all_actions at each line, there is a desc of an action (e.g. [3, 0])
    # we accumulate the index of the action in all_combis
    for ccc, row in enumerate(all_actions):
        print("ccc : {}".format(str(ccc)))
        smth_was_found=False
        for idx, item in enumerate(all_combis):

            if torch.all(torch.tensor(row) == torch.tensor(item)):
                indices.append(idx)
                smth_was_found=True
                break
        if not smth_was_found:
            print("therow of element {}".format(str(ccc)))
            print(row)

    actions_indexes = torch.tensor(indices)

    print(actions_indexes)
    print("len(actions_indexes)")
    print(len(actions_indexes))
    

    # print(type(pre_configs))
    # print("preconfig")
    # print(pre_configs[10])
    # print()
    # print("succonfig")
    # print(suc_configs[10])
    # print()
    # print("action's index")
    # print(actions_indexes[10])
    # print("i.e., action: ")
    # print(all_combis[int(actions_indexes[10])])

    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_combis))

    # print(actions_one_hot[10])
    # print(all_combis[:2])
    # print(type(all_combis))


    # all_actions_tensor = torch.tensor(all_actions)
    # #exit()
    # target_tensor = torch.tensor([0, 2, 0, 9])
    # matches = torch.all(all_actions_tensor == target_tensor, dim=1)
    # print("occurences")
    # occurrences = matches.sum().item()
    # print(occurrences) # 

    
    actions_summed = torch.sum(actions_one_hot, dim=0)
    print(actions_summed)
    print(len(actions_summed))
    #exit()

    pres = p.generate(pre_configs, disks, towers)
    sucs = p.generate(suc_configs, disks, towers)


    
    parameters["picsize"] = pres[0].shape

    print("theconfs")
    print(pre_configs[0]) # [ white red green blue] <=> [1 2 0 3]
    print(suc_configs[0]) # du sampling d'action c'est quoi, c'est
    # pour chaque action, color depart | from | to | color arrivee
    #
    #  par ex, pour action: [ 0 0 1 1 ]  =  un white sur le 0 qu'on met sur le 1 où ya déjà un 1
    #                                         [ 0 1 1 1]

    # blue(3) green(2) red(1) white (0): du plus gros au plus petit


    # plt.imshow(pres[0])
    # plt.savefig('hanoiPRE.png')
    # plt.imshow(sucs[0])
    # plt.savefig('hanoiSUCC.png')


    # converted = []

    # for i in pres[0]:
    #     i


    print("all_combis")
    print(all_combis[128])
    assert len(pres.shape) == 4

    transitions, states = normalize_transitions(pres, sucs)

    # for hh in range(10, 20):
    #     print("for h = {}".format(str(hh)))
    #     #print(np.where(actions_one_hot.numpy()[hh] == 1)[0])
    #     print(all_combis[np.where(actions_one_hot.numpy()[hh] == 1)[0][0]])

    return transitions, actions_one_hot.numpy()



#ae = run(os.path.join("samples",common.sae_path), transitions)

