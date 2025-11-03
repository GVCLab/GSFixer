import torch


class RegSet:
    def __init__(self):
        self.data = []
        self.access_indices = None

    def __len__(self):
        return len(self.data)

    def populate(self, reg_data):
        self.data.extend(reg_data)
        indices = torch.randperm(len(self.data))
        self.access_indices = iter(indices)

    def clear(self):
        self.data = []
        self.access_indices = None

    def __next__(self):
        if len(self.data) == 0:
            raise StopIteration
        try:
            idx = next(self.access_indices)
        except StopIteration:
            indices = torch.randperm(len(self.data))
            self.access_indices = iter(indices)
            idx = next(self.access_indices)
        return self.data[idx]
