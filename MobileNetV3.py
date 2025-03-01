import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=8, min_value=None):
    """确保所有层的通道数都能被除数整除（硬件优化）"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # 确保调整幅度不超过10%
        new_v += divisor
    return new_v

class HSwish(nn.Module):
    """Hard Swish激活函数"""
    def forward(self, x):
        return x * F.hardsigmoid(x)  # 或使用torch.nn.Hardswish()

class SqueezeExcite(nn.Module):
    """SE注意力模块"""
    def __init__(self, in_ch, reduction_ratio=4):
        super().__init__()
        reduced_dim = _make_divisible(in_ch // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_ch, 1),
            HSwish()
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

class ConvBN(nn.Sequential):
    """卷积 + BN + 激活组合"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activation=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            activation(inplace=True) if activation else nn.Identity()
        )

class InvertedResidual(nn.Module):
    """倒置残差块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, exp_ratio, use_se, activation):
        super().__init__()
        self.use_res_connect = (stride == 1) and (in_ch == out_ch)
        exp_ch = _make_divisible(in_ch * exp_ratio)
        
        layers = []
        # 扩展层
        if exp_ch != in_ch:
            layers.append(ConvBN(in_ch, exp_ch, 1, activation=activation))
        
        # 深度卷积
        layers.extend([
            ConvBN(exp_ch, exp_ch, kernel_size, stride=stride, groups=exp_ch, activation=activation),
            SqueezeExcite(exp_ch) if use_se else nn.Identity(),
            # 投影层
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.block = nn.Sequential(*layers)
        self.act = activation(inplace=True) if self.use_res_connect else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return self.act(x + self.block(x))
        return self.block(x)

class MobileNetV3(nn.Module):
    def __init__(self, mode='large', num_classes=1000, width_mult=1.0):
        super().__init__()
        # 配置参数（Large版本）
        if mode == 'large':
            cfg = [
                # 扩张比, 输出通道, 是否SE, 激活函数, 重复次数, stride
                [1, 16,  False, nn.ReLU,   1, 1],
                [4, 24,  False, nn.ReLU,   2, 2],
                [3, 40,  True,  nn.ReLU,   3, 2],
                [6, 80,  False, HSwish,    3, 2],
                [2.5, 112, True,  HSwish,   3, 1],
                [2.3, 160, True,  HSwish,   1, 2],
                [2.3, 160, True,  HSwish,   2, 1],
                [1, 320, True,  HSwish,   1, 1]
            ]
        else:  # small版本配置不同
            # ...（类似结构，参数不同）
            pass

        # 构建第一层
        input_ch = _make_divisible(16 * width_mult)
        self.features = [ConvBN(3, input_ch, 3, stride=2, activation=HSwish)]
        
        # 构建中间层
        for exp_ratio, out_ch, use_se, act, repeats, stride in cfg:
            output_ch = _make_divisible(out_ch * width_mult)
            for i in range(repeats):
                self.features.append(InvertedResidual(
                    in_ch=input_ch,
                    out_ch=output_ch,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    exp_ratio=exp_ratio,
                    use_se=use_se,
                    activation=act
                ))
                input_ch = output_ch
        
        # 构建最后几层
        last_conv_out = _make_divisible(1280 * width_mult)
        self.features.extend([
            ConvBN(input_ch, last_conv_out, 1, activation=HSwish),
            nn.AdaptiveAvgPool2d(1)
        ])
        self.features = nn.Sequential(*self.features)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out, num_classes),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    

if __name__ == "__main__":
    # 创建模型
    model = MobileNetV3(mode='large', num_classes=1000)

    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # torch.Size([1, 1000])