import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from PIL import Image
import random
import pickle
import operator
random.seed(1)
np.random.seed(1)

def generate(ics, gcs, render_fn):
    inits = render_fn(ics)
    goals = render_fn(gcs)
    print("washere000")
    for i,(init,goal) in enumerate(zip(inits,goals)):
        #d = "{}/{}/{:03d}-{:03d}".format(output_dir,name,steps,i)
        d='/workspace/latplanRealOneHotActionsV2/'
        print(d)
        print("washere1")
        print(init.shape)
        print(init[15:40][10:])

        # init = Image.fromarray(init)
        # init = init.convert("L")
        # goal = Image.fromarray(goal)
        # goal = goal.convert("L")

        imageio.imsave(os.path.join(d,"initTTT.png"),init)
        imageio.imsave(os.path.join(d,"goalLLL.png"),goal)



def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_image(a,name):
    #plt.figure(figsize=(6,6))
    plt.figure()
    #plt.imshow(a,interpolation='nearest',cmap='gray',)
    plt.imshow(a)
    plt.savefig(name)

def load_mnist():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)

    # Extract the data and labels from the DataLoader
    x_train, y_train = next(iter(trainloader))
    x_test, y_test = next(iter(testloader))

    x_train = x_train.view(x_train.shape[0], 28, 28)
    x_test = x_test.view(x_test.shape[0], 28, 28)

    return (x_train, y_train), (x_test, y_test)

def mnist(labels=range(10)):
    (x_train, y_train), (x_test, y_test) = load_mnist()
  
    # Convert to float32 and normalize within the range [-1, 1]
    x_train = x_train.float() / 255.0
    x_test = x_test.float() / 255.0
    
    # Flatten the images
    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)

    def select(x, y):
        mask = torch.tensor([label in labels for label in y], dtype=torch.bool)
        return x[mask], y[mask]

    x_train, y_train = select(x_train, y_train)
    x_test, y_test = select(x_test, y_test)
  
    return x_train, y_train, x_test, y_test

# # Example usage
# x_train, y_train, x_test, y_test = mnist()


def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min())

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    print("testttt1")
    image = image.astype(float)
    image = equalize(image)
    image = normalize(image)
    image = enhance(image)
    return image


def return_panels(width, height):

    setting = {}
    setting['base'] = 16

    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    

    x_train, y_train, _, _ = mnist()
    
    filters = [ np.equal(i,y_train) for i in range(9) ]
    imgs    = [ x_train[f] for f in filters ]

    
    panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]
    panels[8] = imgs[8][3].reshape((28,28))
    panels[1] = imgs[1][3].reshape((28,28))

    # print(panels[0])
    # print()
    # print(panels[0].numpy())
    # panels[8] = imgs[8][3].reshape((28,28))
    # panels[1] = imgs[1][3].reshape((28,28))

    panels = np.array([ resize(panel.numpy(), (setting['base'],setting['base'])) for panel in panels])

    panels = preprocess(panels)


    return panels



# with:
# mnist
# 3
# 3
# 5000
# kwargs = {'mode': 'learn', 'aeclass': 'CubeSpaceAE_AMA4Conv', 'comment': 'kltune2', 'hash': '05-06T11:21:55.052'}



#load_puzzle("mnist", 3, 3, 5000, objects=True, {'mode': 'learn', 'aeclass': 'CubeSpaceAE_AMA4Conv', 'comment': 'kltune2', 'hash': '05-06T11:21:55.052'})
#



# All latplan imports
import sys
#sys.path.append(r"..latplanClonedLossTerm/latplan")
sys.path.insert(0, '/workspace/latplanClonedEnforce')
import latplan
from latplan.puzzles.objutil import image_to_tiled_objects, tiled_bboxes
from latplan.main.normalization import normalize_transitions, normalize_transitions_objects
sys.path.remove('/workspace/latplanClonedEnforce')
import json

import torch.nn.functional as F


with open("/workspace/latplanDatasetGen/aux.json","r") as f:
    parameters = json.load(f)["parameters"]

#(parameters)

def int_to_binary(n, max_bits):
    """Convert an integer to its binary representation with a fixed number of bits."""
    if n >= 2**max_bits:
        raise ValueError(f"Number {n} can't be represented with {max_bits} bits.")

    bin_str = bin(n)[2:]  # Convert the integer to binary and remove the '0b' prefix

    return [int(bit) for bit in bin_str.zfill(max_bits)]


from itertools import permutations 
def all_orderings():
    # Adjusting the list to [1, 2, 3, 4, 5, 6, 7, 8] and creating all possible orderings
    adjusted_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    all_orderings_adjusted = list(permutations(adjusted_list))
    #print(all_orderings_adjusted)
    #exit()
    return all_orderings_adjusted

def swapPositions(liste, pos1, pos2):

    # print("liste: ")
    # print(liste)

    liste1 = liste.copy()

    # print(liste1)
    # print("liste1")
    
    liste1[pos1], liste1[pos2] = liste[pos2], liste[pos1]
    #print(liste1)
    
    return liste1


def return_all_next_states(config):
    # config is list of size 9
    # config[0] is the digit on the top left corner
    # config[8] is the digit on the bottom right corner

    all_next_states = []

    # where is 0 ?
    index_zero = config.index(0)

    # if index_zero is 0
    if index_zero == 0:
        # switch with pos1 and pos3
        list1 = swapPositions(config, index_zero, 1)
        list2 = swapPositions(config, index_zero, 3)
        all_next_states.append(list1)
        all_next_states.append(list2)

    # elif index_zero is 1
    if index_zero == 1:
        # switch with pos0 and pos2 and pos4
        list1 = swapPositions(config, index_zero, 0)
        list2 = swapPositions(config, index_zero, 2)
        list3 = swapPositions(config, index_zero, 4)
        all_next_states.append(list1)
        all_next_states.append(list2)
        all_next_states.append(list3)

    # elif index_zero is 2
    if index_zero == 2:
        # switch with pos1 and pos5
        list1 = swapPositions(config, index_zero, 1)
        list2 = swapPositions(config, index_zero, 5)
        all_next_states.append(list1)
        all_next_states.append(list2)

    # elif index_zero is 3
    if index_zero == 3:
        # switch with pos0 and pos4 and pos6
        list1 = swapPositions(config, index_zero, 0)
        list2 = swapPositions(config, index_zero, 4)
        list3 = swapPositions(config, index_zero, 6)
        all_next_states.append(list1)
        all_next_states.append(list2)
        all_next_states.append(list3)

    # elif index_zero is 4
    if index_zero == 4:
        # switch with pos1, pos3, pos5, pos7
        list1 = swapPositions(config, index_zero, 1)
        list2 = swapPositions(config, index_zero, 3)
        list3 = swapPositions(config, index_zero, 5)
        list4 = swapPositions(config, index_zero, 7)
        all_next_states.append(list1)
        all_next_states.append(list2)
        all_next_states.append(list3)
        all_next_states.append(list4)

    # elif index_zero is 5
    if index_zero == 5:
        # switch with pos2, pos4, pos8
        list1 = swapPositions(config, index_zero, 2)
        list2 = swapPositions(config, index_zero, 4)
        list3 = swapPositions(config, index_zero, 8)
        all_next_states.append(list1)
        all_next_states.append(list2)
        all_next_states.append(list3)

    # elif index_zero is 6
    if index_zero == 6:
        # switch with pos3, pos7
        list1 = swapPositions(config, index_zero, 3)
        list2 = swapPositions(config, index_zero, 7)
        all_next_states.append(list1)
        all_next_states.append(list2)

    # elif index_zero is 7
    if index_zero == 7:
        # switch with pos6, pos4, pos8
        list1 = swapPositions(config, index_zero, 6)
        list2 = swapPositions(config, index_zero, 4)
        list3 = swapPositions(config, index_zero, 8)
        all_next_states.append(list1)
        all_next_states.append(list2)
        all_next_states.append(list3)

    # elif index_zero is 8
    if index_zero == 8:
        # switch with pos5, pos7
        list1 = swapPositions(config, index_zero, 5)
        list2 = swapPositions(config, index_zero, 7)
        all_next_states.append(list1)
        all_next_states.append(list2)

    return all_next_states





def return_trace_from_one_config(config, trace_length):

    final_length = 0

    all_pairs = []

    last_two_states = []

    while final_length < trace_length:

        next_states = return_all_next_states(config)

        if last_two_states == 2:
            while True:
                next_state = random.choice(next_states)

                if next_state != last_two_states[0]:
                    break
        else:
            next_state = random.choice(next_states)

        last_two_states.append([config, next_state])
        if len(last_two_states) > 2:
            last_two_states.pop(0)

        all_pairs.append([config, next_state])

        config = next_state

        final_length += 1

    return all_pairs

def return_all_transitions_configs():
    all_states = all_orderings()

    # print(all_states)
    # exit()
    all_transitions = []

    for s in all_states:
        next_states = return_all_next_states(list(s))
        for n in next_states:
            all_transitions.append([list(s), n])

    return all_transitions

# fonction qui retourne les actions à partir de config
def from_pairs_of_confis_to_onehotaction_repr(pres_configs, sucs_configs, augmented=False, custom=False, shuffle_dataset=None):


    ##################################################################
    # determining the Symbolic repr of the actions (l/r/t/d and pos0)
    ##################################################################
    
    tensor_pres_configs = torch.tensor(pres_configs)
    
    one_hot_repr = F.one_hot(tensor_pres_configs.to(torch.int64), num_classes=3*3) # all items of pres_configs turned into one-hot
    if not custom:
        permuted_tensor = one_hot_repr.permute(0, 2, 1) # TO USE FOR DEFAULT DATASET 
    else:
        permuted_tensor = one_hot_repr.permute(0, 1, 2) # TO USE FOR CUSTOM DATASET 
    right_pres_configs = permuted_tensor.argmax(dim=2) # see just below


    # right_pres_configs[0] = tensor([4, 6, 7, 3, 5, 1, 2, 0, 8])
    #      = configuration of example 0 (of the pres), in the good order (from top left to bottom right)

    tensor_sucs_configs = torch.tensor(sucs_configs)


    one_hot_repr = F.one_hot(tensor_sucs_configs.to(torch.int64), num_classes=3*3)
    if not custom:
        permuted_tensor = one_hot_repr.permute(0, 2, 1) # TO USE FOR DEFAULT DATASET 
    else:
        permuted_tensor = one_hot_repr.permute(0, 1, 2) # TO USE FOR CUSTOM DATASET
    right_sucs_configs = permuted_tensor.argmax(dim=2)


    # for each pair, determine the pos of the 0
    zero_positions_pres = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_pres_configs])
    zero_positions_sucs = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_sucs_configs])


    ## over all pairs, determine the possible position of 0 in the next state
    pres_minus_three = zero_positions_pres - 3 # returns the position of the zero IF top move
    pres_plus_three = zero_positions_pres + 3  # returns the position of the zero IF down move
    pres_plus_one = zero_positions_pres + 1    # returns the position of the zero IF right move
    pres_minus_one = zero_positions_pres - 1   # returns the position of the zero IF left move


    # just a zeros' vector of size (5000, 1)
    all_actions = torch.zeros_like(zero_positions_pres, dtype=torch.int)


    top_moves = pres_minus_three == zero_positions_sucs # at each index of the dataset says if TOP MOVE true (or false)
    down_moves = pres_plus_three == zero_positions_sucs # ETC
    right_moves = pres_plus_one == zero_positions_sucs
    left_moves = pres_minus_one == zero_positions_sucs

    # actions: 0: top, 1: down, 2: left, 3: right
    # on a 0s vector (all_actions) applies 0, 1, 2 or 3 according to the thruth masks above
    all_actions[top_moves] = 0 
    all_actions[down_moves] = 1
    all_actions[left_moves] = 2
    all_actions[right_moves] = 3

    # tensor([8, 5, 6, 3, 0, 4, 1, 2, 7])
    x = zero_positions_pres[:, None]
    neighbours = torch.gather(right_sucs_configs, 1, x).squeeze()


    
    # now we augment the all_actions with the position of the 0
    pos_and_move = torch.stack((zero_positions_pres, all_actions), dim=1)


    if augmented:
        # Concatenating A and B_reshaped along the second axis (dim=1)
        pos_and_move = torch.cat((pos_and_move, neighbours.unsqueeze(1)), dim=1)

        # pos_and_move[0] est [4 0 0] càd tile milieu, push up et un 0 en haut
        # donc neighbours n'est pas bon


    # hold, for each transition pair, the index of the action from the
    # actions' label vector, i.e. from all_combis or all_combis_augmented
    indices = []


    # in pos_and_move at each line, there is a desc of an action (e.g. [3, 0, 5])
    for ccc, row in enumerate(pos_and_move):
        if augmented:
            smth_was_found=False
            for idx, item in enumerate(all_combis_augmented):
                if torch.all(row == torch.tensor(item)):
                    #print("idx = {}".format(str(idx)))
                    #print("ccc : {}".format(str(ccc)))
                    indices.append(idx)
                    smth_was_found=True
                    break
            if not smth_was_found:
                print("therow of element {}".format(str(ccc)))
                print(row)
        else:
            for idx, item in enumerate(all_combis):
                if torch.all(row == torch.tensor(item)):
                    indices.append(idx)
                    break

    

    actions_indexes = torch.tensor(indices)

    if augmented:
        actions_one_hot = F.one_hot(actions_indexes, num_classes=192)
    else:
        actions_one_hot = F.one_hot(actions_indexes, num_classes=24)

    #print(actions_one_hot.shape) # torch.Size([5000, 24]) or torch.Size([5000, 192])
    summed = torch.sum(actions_one_hot, dim=0)

    #actions_one_hot = torch.ones(40320, 1)

    for hh in range(0, 1000, 500):
        print("for h = {}".format(str(hh)))
        #print(np.where(actions_one_hot.numpy()[hh] == 1)[0])
        print(all_combis[np.where(actions_one_hot.numpy()[hh] == 1)[0][0]])

    return actions_one_hot, all_actions, zero_positions_pres



#onehot_actions = from_pairs_of_confis_to_onehotaction_repr(pres_configs_0, sucs_configs_0)




def return_images_from_configs(pres_configs, sucs_configs):

    import importlib
    generator = 'latplan.puzzles.puzzle_{}'.format("mnist")
    parameters["generator"] = generator

    p = importlib.import_module(generator)

    p.setup()
    
    width=3
    height=3
    

    pres = p.states(width, height, pres_configs)[:,:,:,None]
    sucs = p.states(width, height, sucs_configs)[:,:,:,None]
   

    B, H, W, C = pres.shape
    parameters["picsize"]        = [[H,W]]

    transitions, states = normalize_transitions(pres, sucs)

    return transitions







# # [0,1,2,3,4,5,6,7,8]
# # [0,1,2,3,4,5,6,7,8]
# # actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3
# #print(perform_random_action([0,1,2,3,4,5,6,7,8]))

# # 1) just loop and stack configs
# whole_trace = []

# last_conf = [0,1,2,3,4,5,6,7,8]
# whole_trace.append(last_conf)
# for i in range(5):
#     last_conf, _ = perform_random_action(list(last_conf))
#     whole_trace.append(list(last_conf))

# #pres_configs and sucs_configs
# pres_configs_0 = whole_trace[:-1]
# sucs_configs_0 = whole_trace[1:]

# # 2) then from pres_configs and sucs_configs we can gen i) images and ii) onehotactions
# actions_one_hot, all_actions, zero_positions_pres = from_pairs_of_confis_to_onehotaction_repr(pres_configs_0, sucs_configs_0)

# images = return_images_from_configs(pres_configs_0, sucs_configs_0)

# print(actions_one_hot[2])
# #print(images[0])

# print(pres_configs_0[2])
# plot_image(images[2][0], "2PRES.png")


# print(sucs_configs_0[2])
# plot_image(images[2][1], "2SUCS.png")


def load_puzzle(type, width, height, num_examples, objects, parameters, one_hot=False, augmented=False, custom=False, masked='both', shuffle_dataset=None):
    
    import importlib
    generator = 'latplan.puzzles.puzzle_{}'.format(type)
    print(generator)
    #generator = '/workspace/latplanClonedEnforce/latplan.puzzles.puzzle_{}'.format(type)
    parameters["generator"] = generator
    print(generator)

    # 

    #sys.path.append(r"..latplanClonedLossTerm/latplan")
    sys.path.insert(0, '/workspace/latplanClonedEnforce')
    p = importlib.import_module(generator)
    sys.path.remove('/workspace/latplanClonedEnforce')

    print(p)


    # 
    # <module 'latplan.puzzles.puzzle_mnist' from '/workspace/latplanRealOneHotActionsV2/latplan/puzzles/puzzle_mnist.py'>
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    

    # pairs = return_trace_from_one_config([0,1,2,3,4,5,6,7,8], 5000)

    # data = {
    #     "pairs" : pairs,
    # }
    
    # if not os.path.exists("mnist_dataset"):
    #     os.makedirs("mnist_dataset") 
    # filename = "data.p"
    # with open("mnist_dataset"+"/"+filename, mode="wb") as f:
    #     pickle.dump(data, f)

    
    loaded_dataset = load_dataset("/workspace/latplanDatasetGen/mnist_dataset/data.p")

    pairs = loaded_dataset["pairs"]

    #pairs = pairs[:len(pairs)//8]

    ### dico of occurence of unique pairs
    dico_occurences = {}
    print(type(pairs[0][0]))
    #print(type(pairs[0]))
    exit()

    total_dico = {}

    unique_pairs = []

    unique_states = [] # = the "states" array
    initial_state = str(pairs[0][0])
    final_state = str(pairs[-1][1])
    alphabet_dic = {}

    for occi, pair in enumerate(pairs):
        if pair not in unique_pairs:
            dico_occurences[str(pair)] = 1
            unique_pairs.append(pair)
            alphabet_dic[str(pair)] = "a"+str(occi)
            #alphabet.append()
        else:
            dico_occurences[str(pair)] += 1
        if pair[0] not in unique_states:
            unique_states.append(str(pair[0]))

        if pair[1] not in unique_states:
            unique_states.append(str(pair[1]))

    print("unjique pauirs")
    print(len(unique_pairs)) # 3325
    print(dico_occurences)

    all_transitions = []

    # re loop over the pair and associate each pair to its action
    for occi, pair in enumerate(pairs):
        transi = []
        transi.append(str(pair[0]))
        transi.append(alphabet_dic[str(pair)])
        transi.append(str(pair[1]))
        all_transitions.append(transi)
        
    total_dico["alphabet"] = list(alphabet_dic.values())
    total_dico["states"] = unique_states
    total_dico["initial_state"] = initial_state
    total_dico["accepting_states"] = final_state
    total_dico["transitions"] = all_transitions

    with open("exampleBIS.json", 'w') as f:
        json.dump(total_dico, f)


    exit()
    # build  the alphabet

    # sorted(dico_occurences, key=dico_occurences.get, reverse=True)
    # print("la")
    #print(dico_occurences)
    sorted_dict = dict(sorted(dico_occurences.items(), key=operator.itemgetter(1), reverse=False))
    print(sorted_dict)
    # [[0, 7, 5, 3, 1, 2, 8, 4, 6], [3, 7, 5, 0, 1, 2, 8, 4, 6]]

    # 


    exit()

    # i) list all unique states

    # print(len(unique_states))

    # # ii) for each unique state, print out how many edges join it
    # dico_unique_states_edges = {}
    # for st in unique_states:
    #     dico_unique_states_edges[str(st)] = 0
    #     for pair in pairs:
    #         if pair[1] == st:
    #             dico_unique_states_edges[str(st)] += 1

    #sorted(dico_unique_states_edges, key=dico_unique_states_edges.get, reverse=True)
    # print("la")
    # #print(dico_unique_states_edges)
    # sorted_dict = dict(sorted(dico_unique_states_edges.items(), key=operator.itemgetter(1), reverse=True))
    # print(sorted_dict)
    # exit()

    unique_pairs_copy = unique_pairs.copy()

    print("len unique_pairs_copy")
    print(len(unique_pairs_copy))


    list_of_necessary_states = [
        [1,2,3,4,5,6,7,0,8],

        [1,2,3,4,0,6,7,5,8],

        [1,2,3,4,6,0,7,5,8],

        [1,2,3,4,6,8,7,5,0],

        [1,2,3,4,6,8,7,0,5],

        [1,2,3,4,0,8,7,6,5],

        [1,2,3,4,8,0,7,6,5],

        [1,2,3,4,8,5,7,6,0],

        [1,2,3,4,8,5,7,0,6],

        [1,2,3,4,0,5,7,8,6],

        [1,2,3,4,5,0,7,8,6],

        [1,2,3,4,5,6,7,8,0]
    ]



    immma = p.states(width, height, [[1,2,3,4,5,6,7,0,8]], custom=True)[0][:,:,:,None]
    print("immma[0] data")
    print(immma[0].shape)
    print(np.unique(immma[0]))

    plt.imsave("idd.png",np.squeeze(immma[0]))

    image = imageio.imread("idd.png")
    print("loool")
    print(image.shape)

    print(np.unique(image[:,:,0]))

    # plt.figure(figsize=(1, 1), dpi=48)
    # plt.imshow(immma[0], interpolation='nearest', cmap='gray')
    # plt.axis('off')
    # plt.savefig('imageeeEEEee.png', bbox_inches='tight', pad_inches=0)
    # plt.close()



    # #plt.imsave("imageeeee.png",np.squeeze(immma[0]))
    # plt.figure(figsize=(6,6))
    # plt.axis('off')
    # plt.imshow(np.squeeze(immma[0]),interpolation='nearest',cmap='gray',)
    # plt.savefig("imageeeee.png", bbox_inches='tight', pad_inches = 0)


    exit()
    image = image[:,:,:3]
    print(image.shape)
    image = np.where(image[:,:,0] > 128, 1, 0)
    #image = np.where(image[:,:,0] > 0, 1, 0)
    print(image.shape)
    print(np.unique(image))
    plt.imsave("imageeRECON.PNG",np.squeeze(image))


    
    #immma = p.states(width, height, list_of_necessary_states, custom=True)[0][:,:,:,None]
    #exit()

    for thein, immmm in enumerate(immma):
            

        plt.imsave("immmm"+str(thein)+".PNG",np.squeeze(immmm))



    data = {
        "conseq_1" : immma[0],
        "conseq_2": immma[1],
    }
    
    if not os.path.exists("images_arrays_problem"):
        os.makedirs("images_arrays_problem") 
    filename = "conseqs.p"
    with open("images_arrays_problem"+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)





    list_of_necessary_transitions = []

    for iijj in range(0, len(list_of_necessary_states)-1, 1):
        list_of_necessary_transitions.append([list_of_necessary_states[iijj], list_of_necessary_states[iijj+1]])


    for pair in list_of_necessary_transitions:
        if pair not in unique_pairs_copy:
            unique_pairs_copy.append(pair)

    print("unique_pairs_copy bis")
    print(len(unique_pairs_copy))


    edge_to_remove = [
        [1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0]
    ]

    # for iimm, imgggg in enumerate(immma):
    #     plt.imsave("Pouz"+str(iimm)+".PNG",np.squeeze(imgggg))

    # exit()

    # then, make the list of transitions from it

    


    couunter = 0


    for iz, pp in enumerate(unique_pairs):

        # if loop over the pair we want to remove
        if pp[0] == edge_to_remove[0] and pp[1] == edge_to_remove[1]:
            #if couunter == 0:
            unique_pairs_copy.pop(iz)
            #plt.imsave("PUZZI-"+str(num)+".PNG",np.squeeze(pres[num]))
            # thepair = p.states(width, height, pp, custom=True)[:,:,:,None]
            # plt.imsave("PUZZZZZ-0.PNG",np.squeeze(thepair[0]))
            # plt.imsave("PUZZZZZ-1.PNG",np.squeeze(thepair[1]))
            #couunter+=1

        # if loop over the mirror transitions
        elif pp[0] == edge_to_remove[1] and pp[1] == edge_to_remove[0]:
            unique_pairs_copy.pop(iz)

    unique_pairs = unique_pairs_copy

    print("uuuu")
    print(len(unique_pairs))


    # [3, 7, 5, 0, 1, 2, 8, 4, 6]

    # iii) take the state with the most states that join it

    # iv) remove one transition <=> to the above and print it as reduced images (init / goal)

    # sauve pairs et actions dans pickle

    all_actions_unique = unique_pairs



    # Assuming 'unique_pairs' is already computed as in the previous example

    # Step 1: Convert 'unique_pairs' to a tuple of tuples for faster lookups
    unique_pairs_hashable = [tuple(map(tuple, pair)) for pair in unique_pairs]

    # Step 2: Create a mapping from each unique pair to its index
    pair_to_index = {pair: index for index, pair in enumerate(unique_pairs_hashable)}

    # Step 3: For each pair in the original 'pairs' array, create a one-hot encoded vector
    one_hot_encoded = []
    with open("actions_indices_zero_top_left_go_right.txt", "w") as file1,  open("all_actions.txt", "w") as file2:
        for ind, pair in enumerate(unique_pairs):

  
            if pair[0][0] == 0 and pair[1][1] == 0:
                file1.write(str(ind) + "\n")
                

            file2.write(str(ind)+" "+str(pair) + "\n")

            # Convert the pair to a hashable format
            pair_hashable = tuple(map(tuple, pair))
            
            # Find the index of this pair in 'unique_pairs'
            index = pair_to_index[pair_hashable]
            
            # Create a one-hot encoded vector for this pair
            one_hot_vector = [0] * len(unique_pairs)
            one_hot_vector[index] = 1
            
            # Add the one-hot vector to the list
            one_hot_encoded.append(one_hot_vector)



    one_hot_encoded = np.array(one_hot_encoded)

    unique_pairs = np.array(unique_pairs)

    pres_configs = unique_pairs[:,0,:]

    sucs_configs = unique_pairs[:,1,:]



    # take all the transitions that reprensent pos0 + move_smwhere 

    #     compute the sum of the transitiosn

    #            check in the one_hot_encod finale vector IF there is also the same number of this type of transitions


    actions_transitions_one_hot = one_hot_encoded

    print("shape actions_transitions_one_hot")
    print(actions_transitions_one_hot.shape)
    # sont utilisés les min et max de chaque partie (pres et sucs)
   
    # 1) vérifier si ce sont les mêmes 

    # 2) 

    init_and_goal = p.states(width, height, edge_to_remove, custom=True)[0][:,:,:,None]
    
    #print(init_and_goal.shape)
    init_and_goal = np.array(init_and_goal)
    print(np.unique(init_and_goal))
    print("unique1")

    # init_and_goal = np.where(init_and_goal[:,:,:] > 0.2, 1, 0)
    # print(init_and_goal.shape)
    # exit()

    print(pres_configs.shape)


    # data = {
    #     "init" : init_and_goal[0],
    #     "goal": init_and_goal[1],
    # }
    
    # if not os.path.exists("images_arrays_problem"):
    #     os.makedirs("images_arrays_problem") 
    # filename = "data.p"
    # with open("images_arrays_problem"+"/"+filename, mode="wb") as f:
    #     pickle.dump(data, f)




    plt.imsave("iniit.PNG",np.squeeze(init_and_goal[0]))
    plt.imsave("gooal.PNG",np.squeeze(init_and_goal[1]))

    pres_im, max_, min_ = p.states(width, height, pres_configs, custom=True)
    sucs_im, maxx_, minn_ = p.states(width, height, sucs_configs, custom=True)
    print("luu")
    print(max_)
    print(maxx_)
    print(min_)
    print(minn_)

    pres = pres_im[:,:,:,None]
    sucs = sucs_im[:,:,:,None]



    # pres, _, _ = p.states(width, height, pres_configs, custom=True)[:,:,:,None]
    # sucs, _, _ = p.states(width, height, sucs_configs, custom=True)[:,:,:,None]

    
    # besoin des id et des actions

    print("pres[0]")
    print(pres[0])
    with open("actions_indices_zero_top_left_go_right.txt", "r") as file:
        numbers_list = [int(line.strip()) for line in file]

    # for num in numbers_list:
    #     plt.imsave("PUZZI-"+str(num)+".PNG",np.squeeze(pres[num]))









    if objects:
        pres = image_to_tiled_objects(pres, p.setting['base'])
        sucs = image_to_tiled_objects(sucs, p.setting['base'])
        bboxes = tiled_bboxes(B, height, width, p.setting['base'])
        pres = np.concatenate([pres,bboxes], axis=-1)
        sucs = np.concatenate([sucs,bboxes], axis=-1)
        transitions, states = normalize_transitions_objects(pres,sucs,new_params=parameters)
    
    else:

        transitions, states, mean_, std_ = normalize_transitions(pres, sucs)


    print(np.unique(transitions))
    print(np.min(transitions))
    print(np.max(transitions))



    # on va faire 
    # un train_set et un val_test_set


    train_set = []
    for ip, tran in enumerate(transitions):
        train_set.append([tran, actions_transitions_one_hot[ip]])

    val_test_set = train_set.copy()
    random.shuffle(val_test_set)



    # data = {
    #     "train_set" : train_set,
    #     "val_test_set": val_test_set,
    #     "actions_transitions_one_hot": actions_transitions_one_hot,
    #     "all_actions_unique": all_actions_unique, 
    #     "mean_": mean_, 
    #     "std_": std_
    # }
    
    # if not os.path.exists("puzzle_dataset"):
    #     os.makedirs("puzzle_dataset") 
    # filename = "data.p"
    # with open("puzzle_dataset"+"/"+filename, mode="wb") as f:
    #     pickle.dump(data, f)



    return  train_set, val_test_set[:len(val_test_set)//2], actions_transitions_one_hot, all_actions_unique, mean_, std_, maxx_, minn_




def return_transitions():

    train_set, val_test_set, actions_transitions, states, masks, masked_transitions = load_puzzle("mnist", 3, 3, 5000, False, parameters, one_hot=False)

    return train_set, val_test_set, actions_transitions



def return_transitions_one_hot(augmented=False, custom=False, masked='both', shuffle_dataset=False):

    train_set, val_test_set, actions_transitions, all_actions_unique, mean_, std_, max_, min_ = load_puzzle("mnist", 3, 3, 5000, False, parameters,  one_hot=True, augmented=augmented, custom=custom, masked=masked, shuffle_dataset=shuffle_dataset)

    return train_set, val_test_set, actions_transitions, all_actions_unique, mean_, std_, max_, min_




return_transitions_one_hot(augmented=False, custom=True, masked=None, shuffle_dataset=True)


#print(return_all_next_states([0,1,2,3,4,5,6,7]))

#print(len(return_all_transitions_configs()))

#print(return_trace_from_one_config([0,1,2,3,4,5,6,7,8], 5))