import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import MNISTModule as MM
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset



num_examples=5000



import json
with open("aux.json","r") as f:
    parameters = json.load(f)["parameters"]



class CustomDataset(Dataset):
    def __init__(self, transitions, actions_transitions):
        assert len(transitions) == len(actions_transitions), "Both input tuples must have the same length"
        self.transitions = transitions
        self.actions_transitions = actions_transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        # Combine the transition and its corresponding action_transition
        sample = (self.transitions[idx], self.actions_transitions[idx])
        return sample




class MNISTDataModule(pl.LightningDataModule):


    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    import subprocess
    import sys

    def download_file(self, url, output_filename):
        try:
            # Using curl to download the file
            command = ["curl", "-L", "-o", output_filename, url]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Error downloading {url}")
            sys.exit(1)

    def extract_tar_using_cmd(self, tar_path, dest_dir):
        """Extracts a tar file to a specified directory using the tar command."""
        try:
            command = ["tar", "-xvf", tar_path, "-C", dest_dir]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Error extracting {tar_path}")


    def file_exists_with_extension(self, directory, extension):
        """Check if any file in the directory has the specified extension."""
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                return True
        return False


    def remove_files_with_pattern(self, directory, pattern=r".*\.tar$"):
        for filename in os.listdir(directory):
            if re.match(pattern, filename):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Removed {file_path}")



    def image_to_tiled_objects(self, x, tilesize):
        B, H, W, C = x.shape
        sH, sW = H//tilesize, W//tilesize

        x = x.reshape([B, sH, tilesize, sW, tilesize, C])
        x = np.swapaxes(x, 2, 3) # [B, sH, sW, tilesize, tilesize, C]
        x = x.reshape([B, sH*sW, tilesize*tilesize*C])
        return x




    def tiled_bboxes(self, batch, height, width, tilesize):
        x1 = np.tile(np.arange(width),height)   # [9] : 0,1,2, 0,1,2, 0,1,2
        y1 = np.repeat(np.arange(height),width) # [9] : 0,0,0, 1,1,1, 2,2,2
        x2 = x1+1
        y2 = y1+1
        bboxes = \
            np.repeat(                                     # [B,9,4]
                np.expand_dims(                            # [1,9,4]
                    np.stack([x1,y1,x2,y2],axis=1) * tilesize, # [9,4]
                    0),
                batch, axis=0)
        # [batch, objects, 4]
        return bboxes



    def normalize(self, x,save=True):
        if ("mean" in parameters) and save:
            mean = np.array(parameters["mean"][0])
            std  = np.array(parameters["std"][0])
        else:
            mean               = np.mean(x,axis=0)
            std                = np.std(x,axis=0)
            if save:
                parameters["mean"] = [mean.tolist()]
                parameters["std"]  = [std.tolist()]
        print("normalized shape:",mean.shape,std.shape)
        return (x - mean)/(std+1e-20)


    def normalize_transitions(self, pres,sucs):
        """Normalize a dataset for image-based input format.
    Normalization is performed across batches."""
        B, *F = pres.shape
        transitions = np.stack([pres,sucs], axis=1) # [B, 2, F]
        print(transitions.shape)

        normalized  = self.normalize(np.reshape(transitions, [-1, *F])) # [2BO, F]
        states      = normalized.reshape([-1, *F])
        transitions = states.reshape([-1, 2, *F])
        return transitions, states




    def prepare_data(self):

        import requests
        import tarfile
        import os

        # Download the tar file into "data" dir if not already exists
        if not self.file_exists_with_extension(self.data_dir, ".npz"):
            
            if not self.file_exists_with_extension(self.data_dir, ".tar"):

                self.download_file("https://github.com/guicho271828/latplan/releases/download/v5.0.0/datasets.tar", "data/datasets.tar")

            self.extract_tar_using_cmd("./data/datasets.tar", "./data")

        

    def prepare_data2(self):

        import json
        import os


        generator = 'latplan.puzzles.puzzle_{}'.format("mnist")
        parameters["generator"] = generator

        MM.setup()
        width=3
        height=3
        path = os.path.join("data","","-".join(map(str,["puzzle","mnist",width,height]))+".npz")

        print(path)

        with np.load(path) as data:
            pres_configs = data['pres'][:num_examples]
            sucs_configs = data['sucs'][:num_examples]  



        ##################################################################
        # determining the Symbolic repr of the actions (l/r/t/d and pos0)
        ##################################################################

        tensor_pres_configs = torch.tensor(pres_configs)
        one_hot_repr = F.one_hot(tensor_pres_configs.to(torch.int64), num_classes=3*3)
        permuted_tensor = one_hot_repr.permute(0, 2, 1)
        right_pres_configs = permuted_tensor.argmax(dim=2)


        tensor_sucs_configs = torch.tensor(sucs_configs)
        one_hot_repr = F.one_hot(tensor_sucs_configs.to(torch.int64), num_classes=3*3)
        permuted_tensor = one_hot_repr.permute(0, 2, 1)
        right_sucs_configs = permuted_tensor.argmax(dim=2)

        # for each pair, determine the pos of the 0
        zero_positions_pres = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_pres_configs])
        zero_positions_sucs = torch.tensor([torch.where(vec == 0)[0].item() for vec in right_sucs_configs])

        print("pres")
        print(right_pres_configs[:5])
        print("sucs")
        print(right_sucs_configs[:5])

        # now, for each pair, determine the action's label (l/r/t/p)

        pres_minus_three = zero_positions_pres - 3
        pres_plus_three = zero_positions_pres + 3
        pres_plus_one = zero_positions_pres + 1
        pres_minus_one = zero_positions_pres - 1

        print("zeros positions")
        print(zero_positions_pres[:5])

        all_actions = torch.zeros_like(zero_positions_pres, dtype=torch.int)

        
        top_moves = pres_minus_three == zero_positions_sucs # at each index says if true or false
        down_moves = pres_plus_three == zero_positions_sucs # at each index says if true or false
        right_moves = pres_plus_one == zero_positions_sucs
        left_moves = pres_minus_one == zero_positions_sucs

        # actions: 1: top, 2: down, 3: left, 4: right

        all_actions[top_moves] = 1 
        all_actions[down_moves] = 2
        all_actions[left_moves] = 3
        all_actions[right_moves] = 4

        print("actions")
        print(all_actions[:5])

        # each action for a transition is described by the position of "0" and the type of move (1, 2, 3 or 4)
        # so actions_transitions[0] could be like [7, 2], i.e. initial "0" position is 7 and a down move
        actions_transitions = torch.stack((zero_positions_pres, all_actions), dim=-1)

        print("stacked")
        print(actions_transitions[:5])


    
        ##################################################################
        #                           THE END                              #
        ##################################################################



        pres = MM.states(width, height, pres_configs)[:,:,:,None]
        sucs = MM.states(width, height, sucs_configs)[:,:,:,None]
    


        B, H, W, C = pres.shape
        parameters["picsize"]        = [[H,W]]
        print("loaded. picsize:",[H,W])

    
        transitions, states = self.normalize_transitions(pres, sucs)

        return  transitions, actions_transitions, states




    def setup(self, stage: str):




        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            all_transitions, all_actions_transitions, _ = self.prepare_data2()

            mnist_full = CustomDataset(all_transitions, all_actions_transitions)


            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [int(num_examples*0.95), int(num_examples*0.05)], generator=torch.Generator().manual_seed(42)
            )


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            all_transitions, all_actions_transitions, _ = self.prepare_data2()
            mnist_full = CustomDataset(all_transitions, all_actions_transitions)

            self.mnist_test = random_split(
                mnist_full, num_examples*0.05, generator=torch.Generator().manual_seed(42)
            )

            
    def custom_collate(self, batch):
        
        transitions, actions_transitions = zip(*batch)

        #thestack = torch.stack([torch.tensor(t) for t in transitions], [torch.tensor(a) for a in actions_transitions])

        merged = tuple(zip(transitions, actions_transitions))
    
        return merged

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=400, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=400, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=400, collate_fn=self.custom_collate)


mnist_dataset = MNISTDataModule("./data")

mnist_dataset.prepare_data()

mnist_dataset.setup("fit")

train_dataloader = mnist_dataset.train_dataloader()

train_example = next(iter(train_dataloader))

print(train_example[0])


