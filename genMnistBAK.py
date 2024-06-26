import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from PIL import Image



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

# # Test
# n = 3
# max_bits = 5
# print(int_to_binary(n, max_bits))  # Expected output: [0, 0, 0, 1, 1]




all_combis = []

##  actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3
#

### 0 at the corner ####
# 0 at top left
all_combis.append([0, 1]) # 0
all_combis.append([0, 3]) # 1

# top middle
all_combis.append([1, 1]) # 2
all_combis.append([1, 2]) # 3
all_combis.append([1, 3]) # 4

# 0 at top right
all_combis.append([2, 1]) # 5
all_combis.append([2, 2]) # 6

# left middle
all_combis.append([3, 0]) # 7
all_combis.append([3, 1]) # 8
all_combis.append([3, 3]) # 9

### 0 at the middle
all_combis.append([4, 0]) # 10
all_combis.append([4, 1]) # 11
all_combis.append([4, 2]) # 12
all_combis.append([4, 3]) # 13

# right middle
all_combis.append([5, 0]) # 14
all_combis.append([5, 1]) # 15
all_combis.append([5, 2]) # 16

# 0 at bottom left
all_combis.append([6, 0]) # 17
all_combis.append([6, 3]) # 18

# bottom middle 
all_combis.append([7, 0]) # 19
all_combis.append([7, 2]) # 20
all_combis.append([7, 3]) # 21

# 0 at bottom right
all_combis.append([8, 0]) # 22
all_combis.append([8, 2]) # 23


all_combis_augmented = [] # = the other 8 digit susceptible to be switched, for each action

for combi in all_combis:
    for i in range(1, 9):
        lol = np.copy(combi)
        lol = np.append(lol, i)
        all_combis_augmented.append(lol)


print("all_combis_augmented")
print(all_combis_augmented)
print("all_combis_augmentedlol")

print(all_combis_augmented[41])

print(len(all_combis_augmented)) # 192

##  actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3



# 1) tu combine les 2 vect, donc now (5000, 2) where at index i 0 we habe the pos0 and at i 1 the action_move

# 2) turn this combined vector into a (5000, 24) vector where at index i we have a one-hot repre of pos0 and action_move

# 3) for the one-hot repr we do a one hot repr of the index of value [pos0, actionmove] from all_combis

# from a position of 0 (NO, actually, from a config)
# returns a list of two elements:
# the next pos of 0 and the label's action performed (from 0 to 3)
# the action and the next position are legal
def perform_random_action(config):

    pos0 = config.index(0)

    all_combis = []

    ##  actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3
    #

    ### 0 at the corner ####
    # 0 at top left
    all_combis.append([0, 1])
    all_combis.append([0, 3])

    # top middle
    all_combis.append([1, 1])
    all_combis.append([1, 2])
    all_combis.append([1, 3])

    # 0 at top right
    all_combis.append([2, 1])
    all_combis.append([2, 2])

    # left middle
    all_combis.append([3, 0])
    all_combis.append([3, 1])
    all_combis.append([3, 3])

    ### 0 at the middle
    all_combis.append([4, 0]) # 0 in middle
    all_combis.append([4, 1])
    all_combis.append([4, 2])
    all_combis.append([4, 3])

    # right middle
    all_combis.append([5, 0])
    all_combis.append([5, 1])
    all_combis.append([5, 2])

    # 0 at bottom left
    all_combis.append([6, 0])
    all_combis.append([6, 3])

    # bottom middle 
    all_combis.append([7, 0])
    all_combis.append([7, 2])
    all_combis.append([7, 3])

    # 0 at bottom right
    all_combis.append([8, 0])
    all_combis.append([8, 2])

    all_combis = np.array(all_combis)
    indices = np.where(all_combis[:, 0] == pos0)

    result = all_combis[indices]

    #print(result)

    action_and_next = []

    # actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3
    for app in result:

        if app[1] == 0:
            next_pos = app[0] - 3

        if app[1] == 1:
            next_pos = app[0] + 3

        if app[1] == 2:
            next_pos = app[0] - 1

        if app[1] == 3:
            next_pos = app[0] + 1

        if next_pos >= 0 and next_pos < 9:
            # is valid
            action_and_next.append([next_pos, app[1]])

    import random

    random_item = random.choice(action_and_next)

    nextPos0 = random_item[0]

    new_config = np.copy(np.array(config))

    new_config[pos0], new_config[nextPos0] = config[nextPos0], config[pos0]

    return new_config, random_item[1]

    # from pos0 , prend toutes les actions "applicables" (où pos0 est à gauche)

    # if action==1 (push_down), add 3 to the pos, (if still between 0-9 ok)...
    #    keep this action (in memoty)

    # among all still valid actions (and where they lead to ) choose one randomly

    # return the action label et the next pos0




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
    print("pres_configspres_configspres_configspres_configspres_configspres_configspres_configspres_configspres_configspres_configs")
    
    tensor_pres_configs = torch.tensor(pres_configs)
    
    one_hot_repr = F.one_hot(tensor_pres_configs.to(torch.int64), num_classes=3*3) # all items of pres_configs turned into one-hot
    if not custom:
        permuted_tensor = one_hot_repr.permute(0, 2, 1) # TO USE FOR DEFAULT DATASET 
    else:
        permuted_tensor = one_hot_repr.permute(0, 1, 2) # TO USE FOR CUSTOM DATASET 
    right_pres_configs = permuted_tensor.argmax(dim=2) # see just below

    print("right_pres_configs")
    print(right_pres_configs[0])
    print(right_pres_configs.size())
    

    # right_pres_configs[0] = tensor([4, 6, 7, 3, 5, 1, 2, 0, 8])
    #      = configuration of example 0 (of the pres), in the good order (from top left to bottom right)

    tensor_sucs_configs = torch.tensor(sucs_configs)


    one_hot_repr = F.one_hot(tensor_sucs_configs.to(torch.int64), num_classes=3*3)
    if not custom:
        permuted_tensor = one_hot_repr.permute(0, 2, 1) # TO USE FOR DEFAULT DATASET 
    else:
        permuted_tensor = one_hot_repr.permute(0, 1, 2) # TO USE FOR CUSTOM DATASET
    right_sucs_configs = permuted_tensor.argmax(dim=2)

    print("right_sucs_configs")
    print(right_sucs_configs[0])

    # for each pair, determine the pos of the 0
    zero_positions_pres = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_pres_configs])
    zero_positions_sucs = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_sucs_configs])

    print("zero_positions_pres")
    print(zero_positions_pres[600])

    print()

    print("zero_positions_sucs")
    print(zero_positions_sucs[600])

    print()


    # zero_positions_pres[0] = tensor(7)  (see right_pres_configs[0] above)

    # now, for each pair, determine the action's label (l/r/t/p)

    ## over all pairs, determine the possible position of 0 in the next state
    pres_minus_three = zero_positions_pres - 3 # returns the position of the zero IF top move
    pres_plus_three = zero_positions_pres + 3  # returns the position of the zero IF down move
    pres_plus_one = zero_positions_pres + 1    # returns the position of the zero IF right move
    pres_minus_one = zero_positions_pres - 1   # returns the position of the zero IF left move


    # just a zeros' vector of size (5000, 1)
    all_actions = torch.zeros_like(zero_positions_pres, dtype=torch.int)

    print("pres_minus_three")
    print(pres_minus_three[600]) # tensor([-3, -3, -3])

    print("pres_plus_one")
    print(pres_plus_one[600]) # tensor([1, 1, 1])

    print("pres_minus_one")
    print(pres_minus_one[600])


    top_moves = pres_minus_three == zero_positions_sucs # at each index of the dataset says if TOP MOVE true (or false)
    down_moves = pres_plus_three == zero_positions_sucs # ETC
    right_moves = pres_plus_one == zero_positions_sucs
    left_moves = pres_minus_one == zero_positions_sucs

    print("top_moves[600]")
    print(top_moves[600])

    print("right_moves[600]")
    print(right_moves[600])


    print("left_moves[600]")
    print(left_moves[600])


    print("down_moves[600]")
    print(down_moves[600])

    print("ggggg")
    print(all_actions[600])

    # actions: 0: top, 1: down, 2: left, 3: right
    # on a 0s vector (all_actions) applies 0, 1, 2 or 3 according to the thruth masks above
    all_actions[top_moves] = 0 
    all_actions[down_moves] = 1
    all_actions[left_moves] = 2
    all_actions[right_moves] = 3

    print("theallactions0000000000")
    print(all_actions[600])

    # all_actions is now a (5000, 1) vector WHERE each value at index i is the move (0 1 2 3 for t/d/l/r) that was performed
    # for transition i, e.g. 2 at position 42 means that a left move was performed on transitions number 42 (starting from 0)

    # if the action is 0 (top), look at the digit at position pos0-3
    # if "  "   "  "   1  (down), "  "   "  "   "  "   "  "   pos0+3
    # if "  "   "  "   2  (left), "   "  "   "  "   "  "   "  pos0-1
    # if "  "   "  "   3  (right), "   "  "   "  "   "  "   " pos0+1
    #    for above, look into right_sucs_configs

    print("zero_positions_pres")
    print(zero_positions_pres[0])
    print("iiiii")
    print(right_sucs_configs[0])
    # tensor([8, 5, 6, 3, 0, 4, 1, 2, 7])
    x = zero_positions_pres[:, None]
    neighbours = torch.gather(right_sucs_configs, 1, x).squeeze()

    print(neighbours[0])
    print("neighsssss")

    #neighbours = right_sucs_configs.gather(1, zero_positions_pres.unsqueeze(1)) # shape (5000, 1)

    print("kkk")
    print(all_actions.shape) # 2800
    print(zero_positions_pres.shape)
    print("theallactions")
    print(all_actions[:3]) # [3, 3, 0] DEVRAIT ETRE [3, 3, 1]


    
    # now we augment the all_actions with the position of the 0
    pos_and_move = torch.stack((zero_positions_pres, all_actions), dim=1)


    print("pos_and_move size 1")
    print(pos_and_move.size())
    print(pos_and_move[600])


    if augmented:
        # Concatenating A and B_reshaped along the second axis (dim=1)
        pos_and_move = torch.cat((pos_and_move, neighbours.unsqueeze(1)), dim=1)

        # pos_and_move[0] est [4 0 0] càd tile milieu, push up et un 0 en haut
        # donc neighbours n'est pas bon


    print("pos_and_move size 2")
    print(pos_and_move.size())

    # # TEST to see if number of a combinations is equal to the number of
    # # the <=> one hot label (see below)
    # print("hhhh")
    # target_tensor = torch.tensor([1, 1, 1])
    # occurrences = torch.sum(torch.all(pos_and_move == target_tensor, dim=1)).item()
    # print(occurrences) # 179


    # hold, for each transition pair, the index of the action from the
    # actions' label vector, i.e. from all_combis or all_combis_augmented
    indices = []

    print("ooo")
    print(len(all_combis_augmented)) # 192
    print(len(all_combis)) # 24

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

    print("actions_one_hot")
    print(actions_one_hot) # tensor([[0, 0, 0,  ..., 0, 0, 0],
    print(actions_one_hot.size())

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

    
    print("ppppppppppppp")
    print(p) # <module 'latplan.puzzles.puzzle_mnist' from '/workspace/latplanClonedEnforce/latplan/puzzles/puzzle_mnist.py'>
    # 
    # 
    # <module 'latplan.puzzles.puzzle_mnist' from '/workspace/latplanRealOneHotActionsV2/latplan/puzzles/puzzle_mnist.py'>
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    


    # ics = [(7, 2, 5, 0, 3, 1, 6, 4, 8), (7, 3, 2, 1, 0, 5, 6, 4, 8), (5, 3, 2, 1, 0, 4, 6, 7, 8), (3, 0, 2, 4, 1, 8, 6, 5, 7), (3, 1, 2, 0, 4, 7, 6, 8, 5), (3, 1, 5, 0, 2, 4, 6, 7, 8), (5, 0, 2, 7, 1, 4, 3, 6, 8), (1, 0, 5, 3, 2, 8, 6, 4, 7), (1, 2, 5, 0, 3, 8, 6, 4, 7), (1, 2, 5, 0, 4, 8, 3, 6, 7), (5, 4, 1, 0, 7, 2, 3, 6, 8), (7, 0, 2, 3, 1, 8, 6, 5, 4), (3, 1, 4, 0, 5, 2, 6, 7, 8), (3, 1, 2, 0, 6, 4, 7, 8, 5), (3, 5, 1, 0, 4, 2, 6, 7, 8), (1, 4, 2, 0, 5, 8, 3, 6, 7), (1, 4, 2, 0, 6, 5, 7, 3, 8), (1, 5, 4, 0, 3, 2, 6, 7, 8), (7, 1, 2, 0, 3, 8, 6, 5, 4), (5, 0, 1, 3, 7, 2, 6, 8, 4)]

    # gcs are the goals    

    # ics = [ (7, 2, 5, 4, 3, 1, 6, 0, 8) ]

    # gcs = [ (7, 2, 5, 4, 0, 1, 6, 3, 8) ]

    # generate(ics, gcs, lambda configs: p.generate(np.array(configs),width,height, custom=False))

    # exit()

    # 0 at any pos

    #   then exchange with pos0 + 3 , -3 , +1, -1

    # for each pos of 0
    pres_configs_custom = []
    sucs_configs_custom = []
    

    # pos+ac entre 0 et 9
    # MAIS aussi, si pos=2, alors pos+ac peut pas être 3
    x_and_ys = []
    masks = []
    # +3 uniqument pour de [0,1,2,3,4,5]
    # -3 uniqument pour de [3,4,5,6,7,8]
    # +1 uniqument pour de [0, 1, 3, 4, 6, 7]
    # -1 uniqument pour de [1, 2, 4, 5, 7, 8]
    # pos of the zero
    for pos in range(9):

        # here we only produce pre-states with 0 at center bottom
        # if pos == 7:

        # for each action (+3, -3 etc)
        # for ac in [-3]: # only up
        # down, up, right, left
        # xmin, xmax, ymin, ymax


        

        

        for ac in [3, -3, 1, -1]:

            if ((ac == 3 and 
                pos in [0,1,2,3,4,5]) or
                (ac == -3 and
                pos in [3,4,5,6,7,8]) or
                (ac == 1 and
                pos in [0, 1, 3, 4, 6, 7]) or
                (ac == -1 and
                pos in [1, 2, 4, 5, 7, 8])):


                # DONC AXIS=0 C 'est Y en fait la putain (de haut=0 en bas=48) , AXIS=1 c'est x (de gauche à droite)
                if ac == -1:
                    xmax = ((pos%3) + 1) * 16
                    xmin = xmax - 32
                    ymin = ((pos//3))*16
                    ymax = ymin + 16
                if ac == 1:
                    xmin = ((pos%3)) * 16
                    xmax = xmin + 32
                    ymin = ((pos//3))*16
                    ymax = ymin + 16
                if ac == 3:
                    xmax = ((pos%3) + 1) * 16
                    xmin = xmax - 16
                    ymin = ((pos//3))*16
                    ymax = ymin + 32
                if ac == -3:
                    xmax = ((pos%3) + 1) * 16
                    xmin = xmax - 16
                    ymax = ((pos//3) + 1)*16
                    ymin = ymax - 32
                # 
                print("okboom")
                # for ordering in all_orderings([1, 2, 3, 4, 5, 6, 7, 8]):                        

                #     tmp_pre = np.insert(ordering, pos, 0)
                #     tmp_suc = tmp_pre.copy()
                #     tmp_suc[pos], tmp_suc[pos+ac] = tmp_pre[pos+ac], tmp_pre[pos]
                #     pres_configs_custom.append(tmp_pre)
                #     sucs_configs_custom.append(tmp_suc)


                ##  sample X combinations <=> pos0 + move

                #### ALORS... on va pas sample au hasard

                for s in range(100):

                    prim = np.arange(1, 9)
                    np.random.shuffle(prim)
                    tmp_pre = np.insert(prim, pos, 0)
                    tmp_suc = tmp_pre.copy()
                    tmp_suc[pos], tmp_suc[pos+ac] = tmp_pre[pos+ac], tmp_pre[pos]
                    pres_configs_custom.append(tmp_pre)
                    sucs_configs_custom.append(tmp_suc)

                    thearray = np.zeros((48,48))
                    #thearray[ymin:ymax, xmin:xmax] = 1
                    #thearray[xmin:xmax, ymin:ymax] = 1
                    #thearray[xmax:ymax, xmin:ymin] = 1
                    thearray[ymin:ymax, xmin:xmax] = 1
                    #thearray[0:24, 40:48] = 1 # DONC AXIS=0 C 'est Y en fait la putain (de haut=0 en bas=48) , AXIS=1 c'est x (de gauche à droite)
                    #thearray = np.rot90(thearray)
                    masks.append(thearray)
                    x_and_ys.append([xmin, xmax, ymin, ymax])
    
    print("x_and_ys")
    print(len(x_and_ys))



    import random
    if custom:
        indices = list(range(len(pres_configs_custom)))
        if shuffle_dataset:
            random.shuffle(indices)

        pres_configs_custom_shuffled = [pres_configs_custom[i] for i in indices]
        sucs_configs_custom_shuffled = [sucs_configs_custom[i] for i in indices]

        masks = [masks[i] for i in indices]
        x_and_ys = [x_and_ys[i] for i in indices]

    #           shuffle 100 times the other digits ==> gives the pre, the exchange with the action, gives the sucs
    # 
    with np.load(path) as data:
        pres_configs = data['pres'][:num_examples] # numpy, (5000, 9)
        sucs_configs = data['sucs'][:num_examples]

    if custom:
        print("custom true")
        pres_configs = pres_configs_custom_shuffled
        sucs_configs = sucs_configs_custom_shuffled
   
    # take all the transitions that reprensent pos0 + move_smwhere 

    #     compute the sum of the transitiosn

    #            check in the one_hot_encod finale vector IF there is also the same number of this type of transitions

    #             
   

    actions_one_hot, all_actions, zero_positions_pres = from_pairs_of_confis_to_onehotaction_repr(pres_configs, sucs_configs, augmented=augmented, custom=custom, shuffle_dataset=shuffle_dataset)

    # 
    all_actions_binary_representation = [int_to_binary(action.item(), 2) for action in all_actions]
    # all_actions_binary_representation[:5] = [[1, 1], [1, 0], [0, 1], [1, 0], [0, 0]]

    zero_positions_pres_binary_representation = [int_to_binary(pres.item(), 4) for pres in zero_positions_pres]

    # zero_positions_pres_binary_representation[:5] : [[0, 1, 1, 1], [0, 1, 0, 0], 
    #                                                       [0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]]
    

    # 
    actions_transitions_binary_repr = [a + b for a, b in zip(zero_positions_pres_binary_representation, all_actions_binary_representation)]


    #actions_transitions[:5] = [[0, 1, 1, 1, 1, 1], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
    #                            zero_pos  , action_mov



    actions_transitions_binary_repr = np.array(actions_transitions_binary_repr)



    if one_hot:
        actions_transitions = actions_one_hot.numpy()
    else:
        actions_transitions = actions_transitions_binary_repr

   
    ##################################################################
    #                           THE END                              #
    ##################################################################
    
    # right_sucs_configs
    # pres = p.states(width, height, pres_configs, custom=custom)[:,:,:,None]
    # sucs = p.states(width, height, sucs_configs, custom=custom)[:,:,:,None]

    pres = p.states(width, height, pres_configs)[:,:,:,None]
    sucs = p.states(width, height, sucs_configs)[:,:,:,None]
    



    # !!!!!!
    # config should not be printed from pres_configs or sucs_config here,
    # because these are corrupted, instead they should be found inside the 
    # from_pairs_of_confis_to_onehotaction_repr function
    # then look at the right_pres_configs for instance
    # !!!!!!
    # 2 1 2
    print("jhjh")
    print(len(pres)) # 9600
    
    # plot_image(pres[4501], "4501PRES.png")
    # plot_image(sucs[4501], "4501SUCS.png")
    # print("themassssk")
    # print(x_and_ys[4501]) # [16, 48, 32, 48]

    # #print(masks[4501])
    # plot_image(masks[4501], "4501MASK.png")
    #print(actions_transitions[600])
    print("putain")
    print(np.argmax(np.array(actions_transitions[555])))


    print('the f*** action')
    aaaa = np.array(actions_transitions[11]).squeeze()
    print(np.where(aaaa == 1))
    #exit()
    print(all_combis_augmented[8])
    # actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3

    # B, H, W, C = pres.shape
    # parameters["picsize"]        = [[H,W]]
    # print("loaded. picsize:",[H,W])

    print("pres.shape")
    print(pres.shape) # doit etre comme (12000, 48, 48, 1)


    if objects:

        pres = image_to_tiled_objects(pres, p.setting['base'])
        sucs = image_to_tiled_objects(sucs, p.setting['base'])
        bboxes = tiled_bboxes(B, height, width, p.setting['base'])
        pres = np.concatenate([pres,bboxes], axis=-1)
        sucs = np.concatenate([sucs,bboxes], axis=-1)
        transitions, states = normalize_transitions_objects(pres,sucs,new_params=parameters)
    else:

        transitions, states, mean_, std_ = normalize_transitions(pres, sucs)

    print("lolilol")
    print(transitions.shape)
    print(pres.shape)

    #


    if masked == 'both':

        print(pres.shape)
        print(sucs.shape)
        pres_cop = np.copy(pres).squeeze()
        sucs_cop = np.copy(sucs).squeeze()
        absolute_diff = np.abs(pres_cop - sucs_cop) # (?, 48, 48)

        # xmin, xmax
        maxes_from_axis_two = np.amax(absolute_diff, axis=2) # (?, 48)
        xmin = np.argmax(maxes_from_axis_two, axis=1)
        #xmax = xmin + 15
        flipped_axis_one = np.flip(maxes_from_axis_two, axis=1)
        xmax = 48 - np.argmax(flipped_axis_one, axis=1)

        maxes_from_axis_one = np.amax(absolute_diff, axis=1) # (?, 48)
        # ymin, ymax
        ymin = np.argmax(maxes_from_axis_one, axis=1)
        flipped_axis_two = np.flip(maxes_from_axis_one, axis=1)
        ymax = 48 - np.argmax(flipped_axis_two, axis=1)

        # 1s on between the rectangle, 0s elsewhere
        mask = np.zeros(pres_cop.shape)
        
        for i in range(pres_cop.shape[0]):
            mask[i, xmin[i]:xmax[i], ymin[i]:ymax[i]] = 1

        # then elementwise multi of mask with original dataset
        pres = np.multiply(pres_cop, mask)
        sucs = np.multiply(sucs_cop, mask)
        pres = np.expand_dims(pres, axis=-1)
        sucs = np.expand_dims(sucs, axis=-1)
        transitions = np.stack((pres, sucs), axis=1)

        
    # Masking the transitions into "masked"

    print(transitions.shape)

    #print(np.array(masks))
    masks_expanded = np.expand_dims(np.repeat(np.array(masks)[:, np.newaxis, :, :], 2, axis=1), axis=-1)

    print(masks_expanded.shape)

    masked_transitions = np.multiply(transitions, masks_expanded)

    # plot_image(np.squeeze(transitions[4501][0]), "4501PRE.png")
    # plot_image(np.squeeze(transitions[4501][1]), "4501SUC.png")
    # plot_image(np.squeeze(masked_transitions[4501][0]), "4501MASK0.png")
    # plot_image(np.squeeze(masked_transitions[4501][1]), "4501MASK1.png")

    #exit()

    return  transitions, actions_transitions, states, masks, masked_transitions, mean_, std_




def return_transitions():

    transitions, actions_transitions, states, masks, masked_transitions = load_puzzle("mnist", 3, 3, 5000, False, parameters, one_hot=False)

    return transitions, actions_transitions



def return_transitions_one_hot(augmented=False, custom=False, masked='both', shuffle_dataset=False):

    transitions, actions_transitions, states, masks, masked_transitions, mean_, std_ = load_puzzle("mnist", 3, 3, 5000, False, parameters,  one_hot=True, augmented=augmented, custom=custom, masked=masked, shuffle_dataset=shuffle_dataset)

    return transitions, actions_transitions, masks, masked_transitions, mean_, std_



 #return_transitions_one_hot(augmented=False, custom=False, masked='both', shuffle_dataset=False)


#print(return_all_next_states([0,1,2,3,4,5,6,7]))

#print(len(return_all_transitions_configs()))