import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np



class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** (0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** (0.5), use_wscale=False,
                 lrmul=1, bias=True,
                 intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, (1, 1, 1, 1))
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)
        self.x_param_backup = None

    def forward(self, x, latent, latent_after_trans=None):
        if x is not None:
            if latent_after_trans is None:
                style = self.lin(latent)  # style => [batch_size, n_channels*2]
                shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
                style = style.view(shape)  # [batch_size, 2, n_channels, ...]
            else:
                style = latent_after_trans

            self.x_param_backup = [x.size(1), x.dim()]
            x = x * (style[:, 0] + 1.) + style[:, 1]
            return x

        else:
            if self.x_param_backup is None:
                print('error: have intialize shape yet')
            # print('Generating latent_after_trans:')
            style = self.lin(latent)  # style => [batch_size, n_channels*2]
            shape = [-1, 2, self.x_param_backup[0]] + (self.x_param_backup[1] - 2) * [1]
            style = style.view(shape)  # [batch_size, 2, n_channels, ...]
            return style


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


# Upscale and blur layers


class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act)
            ,
            ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))



    def make_mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, 512
        ).cuda()
        mean_latent = super().forward(latent_in).mean(0, keepdim=True)
        mean_latent = mean_latent.unsqueeze(1).expand(-1, 18, -1)
        return mean_latent

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, device, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.avg_latent = avg_latent
        self.device = device
        # self.register_buffer('avg_latent', avg_latent)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(self.device)
        return torch.where(do_trunc, interp, x)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles,
                 activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNorm()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None, latent_after_trans=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            if latent_after_trans is None:
                x = self.style_mod(x, dlatents_in_slice)
            else:
                x = self.style_mod(x, dlatents_in_slice, latent_after_trans)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm,
                 use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(dlatent_size, nf * 16, gain=gain / 4,
                                  use_wscale=use_wscale)  # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, dlatents_in_range, latent_after_trans=None):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        if latent_after_trans is None:
            x = self.epi1(x, dlatents_in_range[:, 0])
        else:
            x = self.epi1(x, dlatents_in_range[:, 0], latent_after_trans[0])  # latent_after_trans is a list

        x = self.conv(x)

        if latent_after_trans is None:
            x1 = self.epi2(x, dlatents_in_range[:, 1])
        else:
            x1 = self.epi2(x, dlatents_in_range[:, 1], latent_after_trans[1])

        return x1, x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise,
                 use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, x, dlatents_in_range, latent_after_trans=None):
        x = self.conv0_up(x)

        if latent_after_trans is None:
            x = self.epi1(x, dlatents_in_range[:, 0])
        else:
            x = self.epi1(x, dlatents_in_range[:, 0], latent_after_trans[0])  # latent_after_trans is a list
        x = self.conv1(x)

        if latent_after_trans is None:
            x1 = self.epi2(x, dlatents_in_range[:, 1])
        else:
            x1 = self.epi2(x, dlatents_in_range[:, 1], latent_after_trans[1])
        return x1, x


class SegSynthesisBlock(nn.Module):
    def __init__(self, prev_channel, current_channel, single_in=False):
        super().__init__()
        self.single_in = single_in
        # self.in_conv = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(current_channel, current_channel, 3, 1, 1),
        #     nn.BatchNorm2d(current_channel),
        #     nn.ReLU(),
        #     nn.Conv2d(current_channel, current_channel, 1),
        #     nn.BatchNorm2d(current_channel)
        # )

        if not single_in:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")

            self.out_conv1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(current_channel + prev_channel, current_channel, 1, 1, 0),
                nn.BatchNorm2d(current_channel)
            )

        self.out_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(current_channel + current_channel, current_channel, 1, 1, 0),
            nn.BatchNorm2d(current_channel)
        )

    def forward(self, x_curr, x_curr2, x_prev=None):

        # x_curr = self.in_conv(x_curr)

        if self.single_in:
            x_middle = x_curr
        else:
            x_prev = self.up(x_prev)
            x_concat = torch.cat([x_curr, x_prev], 1)

            x_middle = self.out_conv1(x_concat)

            x_middle = x_middle + x_curr

        x_concat2 = torch.cat([x_curr2, x_middle], 1)
        x_out = self.out_conv2(x_concat2)
        x_out = x_out + x_curr2
        return x_out


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 num_channels=3,  # Number of output color channels.
                 resolution=512,  # Output resolution.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 use_styles=True,  # Enable style inputs?
                 const_input_layer=True,  # First layer is a learned constant?
                 use_noise=True,  # Enable noise inputs?
                 randomize_noise=True,
                 # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'
                 use_wscale=True,  # Enable equalized learning rate?
                 use_pixel_norm=False,  # Enable pixelwise feature vector normalization?
                 use_instance_norm=True,  # Enable instance normalization?
                 dtype=torch.float32,  # Data type to use for activations and outputs.
                 fused_scale='auto',
                 # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
                 blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
                 structure='auto',
                 # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 is_template_graph=False,
                 # True = template graph constructed by the Network class, False = actual evaluation.
                 force_clean_graph=False,
                 # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
                 seg_branch=False
                 ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        self.seg_branch = seg_branch
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        if is_template_graph: force_clean_graph = True
        if force_clean_graph: randomize_noise = False
        if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        if self.seg_branch:
            seg_block = []
        for res in range(2, resolution_log2 + 1):

            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))



            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                               use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))

                if self.seg_branch:

                    name = '{s}x{s}_seg'.format(s=2 ** res)

                    if len(seg_block) == 0:
                        seg_block.append((name,
                                          SegSynthesisBlock(last_channels, channels, single_in=True)))
                    else:
                        seg_block.append((name,
                                          SegSynthesisBlock(last_channels, channels)))

            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)

        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        if self.seg_branch:
            seg_block.append(("seg_out", nn.Conv2d(channels, 34, 1)))
            self.seg_block = nn.ModuleDict(OrderedDict(seg_block))

    def forward(self, dlatents_in, latent_after_trans=None):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        batch_size = dlatents_in.size(0)
        result_list = []

        if self.seg_branch:
            seg_branch_feature = None
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                if latent_after_trans is None:
                    x, x2 = m(dlatents_in[:, 2 * i:2 * i + 2])
                else:
                    x, x2 = m(dlatents_in[:, 2 * i:2 * i + 2], latent_after_trans[2 * i:2 * i + 2])
            else:

                if latent_after_trans is None:
                    x, x2 = m(x, dlatents_in[:, 2 * i:2 * i + 2])
                else:
                    x, x2 = m(x, dlatents_in[:, 2 * i:2 * i + 2],
                              latent_after_trans[2 * i:2 * i + 2])  # latent_after_trans is a tensor list

                if self.seg_branch:

                    name = '{s}x{s}_seg'.format(s=2 ** (i + 2))

                    curr_seg_block = self.seg_block[name]
                    if seg_branch_feature is None:
                        seg_branch_feature = curr_seg_block(x2, x)
                    else:
                        seg_branch_feature = curr_seg_block(x2, x, x_prev=seg_branch_feature)

            result_list.append(x)
            result_list.append(x2)
        rgb = self.torgb(x)
        if self.seg_branch:
            seg = self.seg_block["seg_out"](seg_branch_feature)
            return rgb, seg, result_list
        return rgb, result_list


#### define discriminator

class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = 4
        self.num_new_features = 1

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features,
                       c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer):
        super().__init__(OrderedDict([
            ('conv0', MyConv2d(in_channels, in_channels, 3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', BlurLayer()),
            ('conv1_down', MyConv2d(in_channels, out_channels, 3, gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class DiscriminatorTop(nn.Sequential):
    def __init__(self, mbstd_group_size, mbstd_num_features, in_channels, intermediate_channels, gain, use_wscale,
                 activation_layer, resolution=4, in_channels2=None, output_features=1, last_gain=1):
        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer(mbstd_group_size, mbstd_num_features)))
        if in_channels2 is None:
            in_channels2 = in_channels
        layers.append(
            ('conv', MyConv2d(in_channels + mbstd_num_features, in_channels2, 3, gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', MyLinear(in_channels2 * resolution * resolution, intermediate_channels, gain=gain,
                                          use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(
            ('dense1', MyLinear(intermediate_channels, output_features, gain=last_gain, use_wscale=use_wscale)))
        super().__init__(OrderedDict(layers))


class D_basic(nn.Sequential):

    def __init__(self,
                 # images_in,                          # First input: Images [minibatch, channel, height, width].
                 # labels_in,                          # Second input: Labels [minibatch, label_size].
                 num_channels=3,  # Number of input color channels. Overridden based on dataloader.
                 resolution=512,  # Input resolution. Overridden based on dataloader.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu',
                 use_wscale=True,  # Enable equalized learning rate?
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
                 mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
                 # blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        self.mbstd_group_size = 4
        self.mbstd_num_features = 1
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        self.gain = gain
        self.use_wscale = use_wscale
        super().__init__(OrderedDict([
                                         ('fromrgb', MyConv2d(num_channels, nf(resolution_log2 - 1), 1, gain=gain,
                                                              use_wscale=use_wscale)),
                                         ('act', act)]
                                     + [('{s}x{s}'.format(s=2 ** res),
                                         DiscriminatorBlock(nf(res - 1), nf(res - 2), gain=gain, use_wscale=use_wscale,
                                                            activation_layer=act)) for res in
                                        range(resolution_log2, 2, -1)]
                                     + [('4x4',
                                         DiscriminatorTop(mbstd_group_size, mbstd_num_features, nf(2), nf(2), gain=gain,
                                                          use_wscale=use_wscale, activation_layer=act))]))