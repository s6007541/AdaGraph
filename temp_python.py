import torch
from torch_geometric.data import Data
# path_ = "Amazon/Photo/processed/data.pt"
# path = "GraphOOD-EERM/data/" + path_
# a, b = torch.load(path)
# a = Data.from_dict(a.__dict__)

# path_ = "Amazon/Photo/processed/data_n.pt"
# path = "GraphOOD-EERM/data/" + path_
# torch.save((a,b), path)

path_ = "Amazon/Photo/processed/data.pt"
path = "GraphOOD-EERM/data/" + path_
a, b = torch.load(path)
print(a)