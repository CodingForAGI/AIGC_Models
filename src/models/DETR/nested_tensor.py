import torch

class NestedTensor:
    def __init__(self, tensors, mask=None):
        # tensors: a list of tensors, where each tensor may have a different shape
        # mask: an optional boolean mask to indicate the valid regions
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device):
        # Move the tensor and mask to the specified device
        cast_tensor = [t.to(device) for t in self.tensors]
        mask = self.mask
        if mask is not None:
            mask = mask.to(device)
        return type(self)(cast_tensor, mask)
    
    def decompose(self):
        # Return the original tensor and the mask
        return self.tensors, self.mask
    
    @classmethod
    def from_tensor_list(cls, tensor_list):
        # Create NestedTensor from a list of tensors
        if tensor_list[0].ndim == 3:
            # Process image data
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            # Handle tensors in other dimensions
            raise ValueError('not supported')
        return cls(tensor, mask)
    
    def __repr__(self):
        return str(self.tensors)
