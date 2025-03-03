import torch
import torch.nn as nn
import torch.nn.functional as F


def image2emb_naive(image, patch_size, weight):
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    patch_embedding = patch @ weight
    return patch_embedding




if __name__ == "__main__":
    bs, ic, image_h, image_v = 16,3,224,224
    patch_size = 16
    model_dim = 8
    max_num_token = 16
    num_classes = 10

    patch_depth = patch_size * patch_size * ic
    image = torch.randn(bs, ic, image_h, image_v)
    weight = torch.randn(patch_depth, model_dim)
    patch_embedding_naive = image2emb_naive(image, patch_size, weight)
    print("shape = ", patch_embedding_naive.shape)