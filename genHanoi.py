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
    pres = p.generate(pre_configs,disks,towers)
    sucs = p.generate(suc_configs,disks,towers)
    assert len(pres.shape) == 4

    transitions, states = normalize_transitions(pres, sucs)

    return transitions



#ae = run(os.path.join("samples",common.sae_path), transitions)

