# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import math
from torch.distributions import Categorical

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, input_dim, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, self.feature_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, self.feature_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, self.feature_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, self.feature_dim, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class ToStyleCode(nn.Module):
    def __init__(self, n_convs, input_dim=512, out_dim=512):
        super(ToStyleCode, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        for i in range(n_convs):
            if i == 0:
                self.convs.append(
                    nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                # self.convs.append(nn.BatchNorm2d(out_dim))
                # self.convs.append(nn.InstanceNorm2d(out_dim))
                self.convs.append(nn.LeakyReLU(inplace=True))
            else:
                self.convs.append(
                    nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                self.convs.append(nn.LeakyReLU(inplace=True))

        self.convs = nn.Sequential(*self.convs)
        self.linear = nn.Linear(out_dim, out_dim)

    def forward(self, x):

        x = self.convs(x)

        x = x.view(-1, self.out_dim)
        x = self.linear(x)
        return x


class ToStyleHead(nn.Module):
    def __init__(self, input_dim=512, out_dim=512):
        super(ToStyleHead, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(
            conv3x3_bn_relu(input_dim, input_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            # output 1x1
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], self.out_dim)
        return x


class FPNEncoder(nn.Module):
    def __init__(self, input_dim, n_latent=14, use_style_head=False, only_last_layer=False, same_view_code=False):
        super(FPNEncoder, self).__init__()
        self.only_last_layer = only_last_layer
        self.n_latent = n_latent
        self.same_view_code = same_view_code
        num_blocks = [3, 4, 6, 3]  # resnet 50
        self.FPN_module = FPN(input_dim, Bottleneck, num_blocks)

        # course block 0-2, 4x4->8x8
        self.course_styles = nn.ModuleList()
        for i in range(3):
            if use_style_head:
                self.course_styles.append(ToStyleHead())
            else:
                self.course_styles.append(ToStyleCode(n_convs=5))
        # medium1 block 3-6 16x16->32x32
        self.medium_styles = nn.ModuleList()
        for i in range(4):
            if use_style_head:
                self.medium_styles.append(ToStyleHead())
            else:
                self.medium_styles.append(ToStyleCode(n_convs=6))
        # fine block 7-13 64x64->256x256
        self.fine_styles = nn.ModuleList()
        for i in range(n_latent - 7):
            if use_style_head:
                self.fine_styles.append(ToStyleHead())
            else:
                self.fine_styles.append(ToStyleCode(n_convs=7))

    def half_forward(self, p2, p3, p4):
        styles = []
        for style_map in self.course_styles:
            styles.append(style_map(p4))

        for style_map in self.medium_styles:
            styles.append(style_map(p3))

        for style_map in self.fine_styles:
            styles.append(style_map(p2))

        if self.only_last_layer:
            last_style = styles[-1]
            styles = last_style.unsqueeze(1).expand(-1, self.n_latent, -1)
        else:
            styles = torch.stack(styles, dim=1)
        return styles


    def forward(self, x):

        styles = []
        # FPN feature
        # import ipdb
        # ipdb.set_trace()
        p2, p3, p4 = self.FPN_module(x)

        count = 0
        for style_map in self.course_styles:
            count +=1
            styles.append(style_map(p4))



        for style_map in self.medium_styles:
            count += 1
            styles.append(style_map(p3))

            if count == 4 and self.same_view_code :

                last_code = styles[-1]
                styles = []
                for i in range(4):
                    styles.append(last_code)


        for style_map in self.fine_styles:
            styles.append(style_map(p2))

        if self.only_last_layer:
            last_style = styles[-1]
            styles = last_style.unsqueeze(1).expand(-1, self.n_latent, -1)
        else:
            styles = torch.stack(styles, dim=1)

        return styles





class FPNEncoder_Disentangle(nn.Module):
    def __init__(self, input_dim, n_latent=14, only_last_layer=False,
                        same_view_code=False, pretrain="", use_flow=False, use_inherit_mask=False, large_mask=False, enforce_thres=0):
        super(FPNEncoder_Disentangle, self).__init__()
        self.identity_encoder = FPNEncoder(input_dim, n_latent=n_latent, only_last_layer=only_last_layer, same_view_code=same_view_code)
        self.content_encoder = FPNEncoder(input_dim, n_latent=n_latent, only_last_layer=only_last_layer,
                                          same_view_code=same_view_code)
        self.use_flow = use_flow
        self.large_mask = large_mask
        self.use_inherit_mask = use_inherit_mask
        self.enforce_thres = enforce_thres
        self.n_latent = n_latent
        if use_flow:
            import sys
            sys.path.append('./flownet2')
            from models import FlowNet2
            self.flow_net = FlowNet2().cuda()
            dict = torch.load("./flownet2/checkpoint/FlowNet2_checkpoint.pth.tar")
            self.flow_net.load_state_dict(dict["state_dict"])
            self.flow_encoder = FPNEncoder(2, n_latent=n_latent, only_last_layer=only_last_layer,
                                              same_view_code=same_view_code)
            if use_inherit_mask:
                self.inherit_mask = nn.Parameter(torch.rand((n_latent, 2), requires_grad=True))
        else:

            if pretrain != "":
                cp = torch.load(pretrain)
                self.identity_encoder.load_state_dict(cp['model_state_dict'], strict=True)
                # self.content_encoder.load_state_dict(cp['model_state_dict'], strict=True)

        if self.large_mask:
            self.mask = nn.Parameter(torch.rand((n_latent * 512, 2), requires_grad=True))
        else:
            self.mask = nn.Parameter(torch.rand((n_latent, 2), requires_grad=True))

    def freeze_mask(self):
        self.mask.requires_grad = False

        for i in self.identity_encoder.parameters():
            i.requires_grad = False
        for i in self.content_encoder.parameters():
            i.requires_grad = False

    def forward(self, x):

        id_code = self.identity_encoder(x)
        content_code = self.content_encoder(x)

        mask = F.softmax(self.mask, dim=-1)
        entropy = Categorical(probs=mask).entropy().mean()

        if self.large_mask:
            mask = mask.view(self.n_latent, 512, 2)
            styles = mask[:, :, 0].unsqueeze(0) * id_code + \
                     mask[:, :, 1].unsqueeze(0) * content_code
            if self.enforce_thres > 0:
                channel_number = mask[:, :, 0].view(-1).shape[0]
                enforce_num = int(channel_number * self.enforce_thres)
                top_values, _ = mask[:, :, 0].view(-1).topk(enforce_num)
                entropy += (1 - top_values).mean() * 10.

            else:
                entropy += 2 * Categorical(probs=mask[:, :, 0].view(-1)).entropy().mean()

            weight_loss =  (1 - mask[:, :, 0].view(-1)).mean()

            entropy += weight_loss

        else:
            import ipdb
            ipdb.set_trace()
            styles = mask[:,0:1].unsqueeze(0) *  id_code + \
                 mask[:,1:2].unsqueeze(0) * content_code

            entropy += 2 * Categorical(probs=mask[:,0]).entropy().mean()


        return styles, id_code, content_code, entropy

    def forward_id(self, x, id_code):

        content_code = self.content_encoder(x)

        mask = F.softmax(self.mask, dim=1)


        if self.large_mask:
            mask = mask.view(self.n_latent, 512, 2)
            styles = mask[:, :, 0].unsqueeze(0) * id_code + \
                     mask[:, :, 1].unsqueeze(0) * content_code
        else:

            styles = mask[:,0:1].unsqueeze(0) *  id_code +  mask[:,1:2].unsqueeze(0) * content_code

        return styles

    def forward_codes(self, id_code, content_code):

        mask = F.softmax(self.mask, dim=1)


        if self.large_mask:
            mask = mask.view(self.n_latent, 512, 2)
            styles = mask[:, :, 0].unsqueeze(0) * id_code + \
                     mask[:, :, 1].unsqueeze(0) * content_code
        else:

            styles = mask[:,0:1].unsqueeze(0) *  id_code +  mask[:,1:2].unsqueeze(0) * content_code

        return styles

    def forward_code_flow(self, base_im, target_im, base_id_code, base_content_code):

        flow_in = torch.cat([target_im.unsqueeze(0), base_im.unsqueeze(0)]).permute(1, 2, 0, 3,
                                                                                      4).cuda() * 255
        flow =  self.flow_net(flow_in)
        delta_content_code = self.flow_encoder(flow)
        inheirt_code  = base_content_code + delta_content_code


        if self.use_inherit_mask:

            curr_content_code = self.content_encoder(target_im)

            content_code_mask = F.softmax(self.inherit_mask, dim=1)
            content_code = content_code_mask[:, 0:1].unsqueeze(0) * inheirt_code + content_code_mask[:, 1:2].unsqueeze(0) * curr_content_code

            entropy = Categorical(probs=content_code_mask).entropy().mean() + Categorical(probs=content_code_mask.T).entropy().mean()

        else:
            content_code = inheirt_code
            entropy = 0

        mask = F.softmax(self.mask, dim=1)
        mask = mask.detach()

        styles = mask[:,0:1].unsqueeze(0) *  base_id_code +  mask[:,1:2].unsqueeze(0) * content_code
        return styles, content_code, entropy






class latentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(latentLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.v_size = hidden_size

        self.project_input = nn.Linear(self.input_dim, self.hidden_size)
        self.project_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.do_pre_v2h = False

        v2h = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 4 * hidden_size)
        ]
        self.v2h = nn.Sequential(*v2h)

        self.upsacle = nn.Linear(hidden_size, input_dim)


        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).cuda(), torch.zeros(bs, self.hidden_size).cuda()

    def forward(self, h, c, input, skip_final=False):
        """
        :param h: prev hidden
        :param c: prev cell
        :param input: input
        :return:
        """
        h_proj = self.project_h(h)

        input_proj = self.project_input(input)
        v2h_input = h_proj * input_proj
        v = self.v2h(v2h_input)
        tmp = v

        # activations
        g_t = tmp[:, 3 * self.hidden_size:].tanh()
        i_t = tmp[:, :self.hidden_size].sigmoid()
        f_t = tmp[:, self.hidden_size:2 * self.hidden_size].sigmoid()
        o_t = tmp[:, 2*self.hidden_size:3 * self.hidden_size].sigmoid()

        c_t = torch.addcmul(c * f_t, i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())
        if skip_final:
            return h_t, c_t
        out =  self.upsacle(h_t)
        return h_t, c_t, out

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, apply_gram_matrix=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        if not apply_gram_matrix:
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        else:
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(gram_matrix(x_vgg[i]), gram_matrix(y_vgg[i].detach()))

        return loss

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

