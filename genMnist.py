import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio


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


##  actions: push_up: 0, push_down: 1, push_left: 2, push_right: 3



# 1) tu combine les 2 vect, donc now (5000, 2) where at index i 0 we habe the pos0 and at i 1 the action_move

# 2) turn this combined vector into a (5000, 24) vector where at index i we have a one-hot repre of pos0 and action_move

# 3) for the one-hot repr we do a one hot repr of the index of value [pos0, actionmove] from all_combis




def load_puzzle(type, width, height, num_examples, objects, parameters, one_hot=False):
    
    import importlib
    generator = 'latplan.puzzles.puzzle_{}'.format(type)
    parameters["generator"] = generator

    #
    p = importlib.import_module(generator)
    #print(p) # <module 'latplan.puzzles.puzzle_mnist' from '/workspace/latplanClonedEnforce/latplan/puzzles/puzzle_mnist.py'>
    
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    

    # 
    with np.load(path) as data:
        pres_configs = data['pres'][:num_examples] # numpy, (5000, 9)
        sucs_configs = data['sucs'][:num_examples]


    ##################################################################
    # determining the Symbolic repr of the actions (l/r/t/d and pos0)
    ##################################################################

    tensor_pres_configs = torch.tensor(pres_configs)
    one_hot_repr = F.one_hot(tensor_pres_configs.to(torch.int64), num_classes=3*3)
    permuted_tensor = one_hot_repr.permute(0, 2, 1)
    right_pres_configs = permuted_tensor.argmax(dim=2)

    # right_pres_configs[0] = tensor([4, 6, 7, 3, 5, 1, 2, 0, 8])
    #      = configuration of example 0 (of the pres), in the good order (from top left to bottom right)

    tensor_sucs_configs = torch.tensor(sucs_configs)
    one_hot_repr = F.one_hot(tensor_sucs_configs.to(torch.int64), num_classes=3*3)
    permuted_tensor = one_hot_repr.permute(0, 2, 1)
    right_sucs_configs = permuted_tensor.argmax(dim=2)

    # for each pair, determine the pos of the 0
    zero_positions_pres = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_pres_configs])
    zero_positions_sucs = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_sucs_configs])

    # zero_positions_pres[0] = tensor(7)  (see right_pres_configs[0] above)


    # now, for each pair, determine the action's label (l/r/t/p)

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

    all_actions[top_moves] = 0 
    all_actions[down_moves] = 1
    all_actions[left_moves] = 2
    all_actions[right_moves] = 3

    # all_actions is now a (5000, 1) vector WHERE each value at index i is the move that was performed
    # for transition i, e.g. 2 at position 42 means that a left move was performed on transitions number 42 (starting from 0)



    pos_and_move = torch.stack((zero_positions_pres, all_actions), dim=1)



    indices = []
    for row in pos_and_move:
        for idx, item in enumerate(all_combis):
            if torch.all(row == torch.tensor(item)):
                indices.append(idx)
                break

    actions_indexes = torch.tensor(indices)


    actions_one_hot = F.one_hot(actions_indexes, num_classes=24)
    # shape (5000, 24)

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
    


    pres = p.states(width, height, pres_configs)[:,:,:,None]
    sucs = p.states(width, height, sucs_configs)[:,:,:,None]
   
    print("config du puzzle")
    print()
    print(pres_configs[2])
    plot_image(pres[2], "2PRES.png")


    print(sucs_configs[2])
    plot_image(sucs[2], "2SUCS.png")

   

    B, H, W, C = pres.shape
    parameters["picsize"]        = [[H,W]]
    print("loaded. picsize:",[H,W])

    if objects:
        pres = image_to_tiled_objects(pres, p.setting['base'])
        sucs = image_to_tiled_objects(sucs, p.setting['base'])
        bboxes = tiled_bboxes(B, height, width, p.setting['base'])
        pres = np.concatenate([pres,bboxes], axis=-1)
        sucs = np.concatenate([sucs,bboxes], axis=-1)
        transitions, states = normalize_transitions_objects(pres,sucs,new_params=parameters)
    else:
        transitions, states = normalize_transitions(pres, sucs)

    return  transitions, actions_transitions, states




def return_transitions():

    transitions, actions_transitions, states = load_puzzle("mnist", 3, 3, 5000, False, parameters, one_hot=False)

    return transitions, actions_transitions



def return_transitions_one_hot():

    transitions, actions_transitions, states = load_puzzle("mnist", 3, 3, 1000, False, parameters,  one_hot=True)

    return transitions, actions_transitions

