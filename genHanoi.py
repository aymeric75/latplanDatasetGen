import numpy as np
import os.path
#from ..util.tuning import parameters
#from .common import *
#from .normalization import normalize_transitions
#from . import common

import sys
#sys.path.append(r"..latplanClonedLossTerm/latplan")
sys.path.insert(0, '/workspace/latplanClonedEnforce')
import latplan
from latplan.puzzles.objutil import image_to_tiled_objects, tiled_bboxes
from latplan.main.normalization import normalize_transitions, normalize_transitions_objects
sys.path.remove('/workspace/latplanClonedEnforce')

import matplotlib.pyplot as plt

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



def return_hanoi_transitions_and_actions(args, parameters):
    parameters["generator"] = "latplan.puzzles.hanoi"
    disks  = args.disks
    towers = args.towers
    num_examples = args.num_examples

    import latplan.puzzles.hanoi as p
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]

    print(type(pre_configs))
    print("preconfig")
    print(pre_configs[:5])
    print()
    print("succonfig")
    print(suc_configs[:5])
    
    diff_between_arrays = pre_configs - suc_configs # still a numpy array
    print()
    # print("diff arrays")
    # print(diff_between_arrays[:5])
    diff_between_arrays_abs = np.absolute(diff_between_arrays)
    # index where not 0
    non_zero_indices = np.argmax(diff_between_arrays_abs, axis=1)
    print()
    # print("indices where change occured")
    # print(non_zero_indices[:5])
    
    retrieved_items_pre_configs = [pre_configs[i, non_zero_indices[i]] for i in range(len(pre_configs))]
    
    retrieved_items_suc_configs = [suc_configs[i, non_zero_indices[i]] for i in range(len(suc_configs))]


    # print()
    # print("elements of pre that was moved")
    # print(retrieved_items_pre_configs[:5])
    # print()
    # print("elements of suc that was moved")
    # print(retrieved_items_suc_configs[:5])
    
    all_simple_actions = np.column_stack((retrieved_items_pre_configs, retrieved_items_suc_configs))
    print(all_simple_actions[:5])

    all_actions_with_prering = np.column_stack((non_zero_indices, all_simple_actions))
    print(all_actions_with_prering[:5])

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

    rings_on_target_ = []
    for li in rings_on_target:
        if li.size == 0:
            rings_on_target_.append(9)
        else:
            rings_on_target_.append(np.min(li))

    print(rings_on_target_[:5])
    #pre_configs


    all_actions_with_prering_and_sucring = np.column_stack((all_actions_with_prering, rings_on_target_))

    print(all_actions_with_prering_and_sucring[:5])



    pres = p.generate(pre_configs, disks, towers)
    sucs = p.generate(suc_configs, disks, towers)

    print("theconfs")
    print(pre_configs[0])
    print(suc_configs[0])


    print(pres[0])
    print(pres[0].shape)
    print(type(pres[0]))
    plt.imshow(pres[0])
    plt.savefig('hanoiPRE.png')

    plt.imshow(sucs[0])
    plt.savefig('hanoiSUCC.png')

    # 


    # print(np.array(pres[0]).shape)
    # print(np.array(pres[1]).shape)

    # print(np.array(sucs[0]).shape)
    # print(np.array(sucs[1]).shape)
    
    # print(pres[0][0][0])
    # print(pres[0][0][0].shape)
    # print(type(pres[0][0][0][0]))

    # converted = []

    # for i in pres[0]:
    #     i




    assert len(pres.shape) == 4

    transitions, states = normalize_transitions(pres, sucs)

    return transitions



#ae = run(os.path.join("samples",common.sae_path), transitions)

