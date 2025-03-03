import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    """将输入转为元组"""
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    """Transformer中的前馈网络,增强特征稳定性和模型泛化能力"""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),#归一化
            nn.Linear(dim, hidden_dim),#线性投影到隐藏维度
            nn.GELU(),#激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),#投影到原始维度
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """dim=768,heads=12,dim_head=64"""
        super().__init__()
        inner_dim = dim_head *  heads #所有注意力头拼接后的总维度
        project_out = not (heads == 1 and dim_head == dim) #判断是否需要投影输出。若头数为1且单头维度等于输入维度，则无需投影（直接输出拼接结果）
        
        self.heads = heads
        
        self.scale = dim_head ** -0.5 # 0.125，缩放因子，用于稳定点积注意力
        self.norm = nn.LayerNorm(dim)# 输入归一化

        self.attend = nn.Softmax(dim = -1)# 注意力权重归一化
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)#将输入通过线性层投影为 Q（查询）、K（键）、V（值），输出维度为 inner_dim * 3（三个矩阵拼接）

        self.to_out = nn.Sequential(#若需要投影，则将多头拼接后的结果映射回原始维度 dim，并应用 Dropout；否则直接输出拼接结果
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)#[16, 197, 768]
        qkv = self.to_qkv(x).chunk(3, dim = -1)#[16, 197, 768]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)#d=768/12=64 [16,12,197,64] 每个头的维度被显式分离

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale#[16,12,197,197] Q和K的点积
        
        attn = self.attend(dots)
        attn = self.dropout(attn)#[16,12,197,197]
        
        out = torch.matmul(attn, v)#[16,12,197,64]
        print("out shape = ", out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')#[16,197,768]
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """depth=12,heads=12,dim_head=3072"""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        """
        image_size=224,输入图像尺寸
        patch_size=16,块尺寸
        num_classes=10，分类数
        dim=768，嵌入后向量维度
        depth=12，编码器层数
        """
        image_height, image_width = pair(image_size)#224,224
        patch_height, patch_width = pair(patch_size)#16,16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)#图像块的数量=196
        patch_dim = channels * patch_height * patch_width#块维度=通道数*块宽度*块高度=768
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),#16 * 196 * 768 批次、块序列、展平特征
            nn.LayerNorm(patch_dim),#归一化最后一个维度
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))#[1,197,768]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape#b=16,n=196,768

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)#[1, 197, 768]，为每个图像块嵌入分类token
        x += self.pos_embedding[:, :(n + 1)]#第1、3维度保留全部，第2维度保留前197个元素,为每个batch加上相同的位置嵌入信息·
        
        x = self.dropout(x)

        x = self.transformer(x)#x shape = [1, 197, 768]
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]#x shape = [1, 768]
        
        x = self.to_latent(x)#x shape = [1, 768]
        x = self.mlp_head(x)#x shape = [1, 10]
        #print("shape= ", x.shape)
        return x

if __name__ == "__main__":
    model = ViT(image_size=224, 
                patch_size=16,
                num_classes=10,
                dim=768, # 嵌入维度 = heads * dim_head
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1,
                emb_dropout=0.1)
    
    img = torch.randn(1, 3, 224, 224)  # 16张224x224的RGB图像
    preds = model(img)
    print(preds.shape)  # 输出形状应为 (16, 10)