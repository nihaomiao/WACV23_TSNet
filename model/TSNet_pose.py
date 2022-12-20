import torch
import torch.nn as nn
from model import networks
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
from torchvision import models
import torch.nn.functional as F
import random


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect', debug=False, normalization=False, addcoords=False):
        assert (n_blocks >= 0)
        super(Encoder, self).__init__()
        self.normalization = normalization
        self.debug = debug
        self.n_layers = 1 + n_downsampling + n_blocks
        self.addcoords = addcoords
        if self.addcoords:
            input_nc += 3
        activation = nn.ReLU(True)

        model = [
            [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [[nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2), activation]]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]]

        if debug:
            for n in range(len(model)):
                setattr(self, 'model' + str(n), nn.Sequential(*model[n]))
        else:
            model_stream = []
            for n in range(len(model)):
                model_stream += model[n]
            self.model = nn.Sequential(*model_stream)

    def forward(self, input):
        if self.addcoords:
            input = self.coord_conv(input)
        if self.debug:
            res = input
            print(res.size())
            for n in range(self.n_layers):
                model = getattr(self, 'model' + str(n))
                res = model(res)
                print(res.size())
            if self.normalization:
                res = F.normalize(res, p=2, dim=1)
            return res
        else:
            output = self.model(input)
            if self.normalization:
                output = F.normalize(output, p=2, dim=1)
            return output

    def coord_conv(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat


class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, n_downsampling=4,
                 norm_layer=nn.InstanceNorm2d, return_fea=False,
                 n_blocks=0):
        super(Decoder, self).__init__()
        self.return_fea = return_fea
        self.n_layers = 1 + n_downsampling + n_blocks
        activation = nn.ReLU(True)
        model = []
        ### resnet blocks
        mult = 2 ** n_downsampling
        self.map_conv = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=(1, 1))
        for i in range(n_blocks):
            model += [[ResnetBlock(ngf * mult, padding_type='reflect', activation=activation, norm_layer=norm_layer)]]
        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                 kernel_size=3, stride=1, padding=0),
                       norm_layer(int(ngf * mult / 2)),
                       activation]]
        model += [[nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]]
        if self.return_fea:
            for n in range(len(model)):
                setattr(self, 'model' + str(n), nn.Sequential(*model[n]))
        else:
            model_stream = []
            for n in range(len(model)):
                model_stream += model[n]
            self.model = nn.Sequential(*model_stream)

    def forward(self, prop_fea, syn_fea):
        input = self.map_conv(torch.cat([prop_fea, syn_fea], dim=1))
        if self.return_fea:
            res = input
            for n in range(self.n_layers - 1):
                model = getattr(self, 'model' + str(n))
                res = model(res)  # return feature
            model = getattr(self, 'model' + str(self.n_layers - 1))
            final = model(res)
            return final, res
        else:
            output = self.model(input)
            return output


class FuseNet(nn.Module):
    def __init__(self, ngf=1024, n_blocks=1,
                 padding_type='reflect',
                 norm_layer=nn.InstanceNorm2d):
        super(FuseNet, self).__init__()
        model = []
        self.n_layers = n_blocks
        activation = nn.ReLU(True)

        for i in range(n_blocks):
            model += [[ResnetBlock(ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]]

        model_stream = []
        for n in range(len(model)):
            model_stream += model[n]
        self.model = nn.Sequential(*model_stream)
        self.conv = nn.Conv2d(ngf, ngf // 2, kernel_size=1)

    def forward(self, src_lbl_fea, tar_lbl_fea):
        input = torch.cat((src_lbl_fea, tar_lbl_fea), dim=1)
        bs, _, h, w = input.shape
        output = self.model(input)
        output = self.conv(output)
        return output


class TSNet(nn.Module):
    def __init__(self, lr=0.0002, beta1=0.5, n_blocks=0,
                 n_source=3,
                 lambda_FML=10.0, lambda_VGG=10.0, lambda_CON=10.0, lambda_GRAD=10.0,
                 is_train=True, getIntermFeat=True, label_nc=5,
                 debug=False, lambda_dec=1.0,
                 addcoords=True,
                 ngf=64, n_downsampling=4,
                 use_mask=True,
                 mean=np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)):
        super(TSNet, self).__init__()
        self.lambda_dec = lambda_dec
        self.n_source = n_source
        self.lr = lr
        self.is_train = is_train
        self.use_mask = use_mask
        self.model_names = ['G', 'D', 'DF']
        self.img_enc = Encoder(input_nc=3 + label_nc, addcoords=addcoords, debug=debug,
                               ngf=ngf, n_downsampling=n_downsampling)
        self.img_enc = networks.init_net(self.img_enc, init_type='normal', init_gain=0.02)
        self.lbl_enc = Encoder(input_nc=label_nc, addcoords=addcoords, n_blocks=0, debug=debug,
                               ngf=ngf, n_downsampling=n_downsampling)
        self.lbl_enc = networks.init_net(self.lbl_enc, init_type='normal', init_gain=0.02)
        self.dec = Decoder(output_nc=3, return_fea=True, n_blocks=n_blocks,
                           ngf=ngf, n_downsampling=n_downsampling)
        self.dec = networks.init_net(self.dec, init_type='normal', init_gain=0.02)
        self.fuse_net = FuseNet(ngf=1024, n_blocks=1)
        self.fuse_net = networks.init_net(self.fuse_net, init_type='normal', init_gain=0.02)
        if self.is_train:
            self.netD = networks.define_D(label_nc + 3, 64, 'basic', 3, 'instance', 'normal', 0.02, getIntermFeat)
            # face discriminator
            self.netDF = networks.define_D(3, 64, 'basic', 3, 'instance', 'normal', 0.02, getIntermFeat)
            self.criterionGAN = networks.GANLoss('lsgan').cuda()
            self.criterionFML = torch.nn.L1Loss().cuda()  # for feature matching loss
            self.criterionVGG = VGGLoss()
            self.optimizer_img_enc = torch.optim.Adam(self.img_enc.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_lbl_enc = torch.optim.Adam(self.lbl_enc.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_dec = torch.optim.Adam(self.dec.parameters(), lr=lr * self.lambda_dec, betas=(beta1, 0.999))
            self.optimizer_fuse_net = torch.optim.Adam(self.fuse_net.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.5 * lr, betas=(beta1, 0.999))
            self.optimizer_DF = torch.optim.Adam(self.netDF.parameters(), lr=0.5 * lr, betas=(beta1, 0.999))
            self.optimizers = []
            # note the order of optimizer is related to the setting of learning rate
            self.optimizers.append(self.optimizer_img_enc)
            self.optimizers.append(self.optimizer_lbl_enc)
            self.optimizers.append(self.optimizer_dec)
            self.optimizers.append(self.optimizer_fuse_net)
            # the final one is D
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_DF)
            self.lambda_FML = lambda_FML
            self.lambda_VGG = lambda_VGG
            self.lambda_CON = lambda_CON
            self.lambda_GRAD = lambda_GRAD
            self.loss_names = ['G', 'G_GAN', 'G_FML', 'G_VGG',
                               'GF', 'GF_GAN', 'GF_FML', 'GF_VGG',
                               'D', 'D_real', 'D_fake',
                               'DF', 'DF_real', 'DF_fake',
                               'grad_G', "warp"]
            for loss_name in self.loss_names:
                setattr(self, "loss_" + loss_name, 0.0)
        self.src_img_list = None
        self.src_lbl_list = None
        self.warp_src_img_list = None
        self.tar_img = None
        self.tar_lbl = None
        self.prev_tar_img = None
        self.prev_tar_lbl = None
        self.rec_tar_img = None
        self.normalized_att_maps = None
        if self.use_mask:
            self.mask_img = torch.from_numpy(-mean).view(1, 3, 1, 1).repeat(1, 1, 256, 256).cuda() / 255.0
            fore_mask = torch.zeros((256, 256), dtype=torch.float32)
            fore_mask[:, 64:192] = 1
            self.fore_mask = fore_mask.view(1, 1, 256, 256).cuda()

    def set_train_input(self, src_img_list, src_lbl_list, src_bbox_list, tar_img, tar_lbl, tar_bbox, use_prev=None):
        if use_prev is None:
            self.src_img_list = [src_img.cuda() / 255.0 for src_img in src_img_list]
        else:
            self.src_img_list = []
            for idx, src_img in enumerate(src_img_list):
                is_prev = use_prev[idx]
                if is_prev:
                    self.src_img_list.append(src_img.cuda())
                else:
                    self.src_img_list.append(src_img.cuda() / 255.0)
        self.src_lbl_list = [src_lbl.cuda() for src_lbl in src_lbl_list]
        self.src_bbox_list = [src_bbox.unsqueeze(dim=1).cuda() for src_bbox in src_bbox_list]
        self.tar_img = tar_img.cuda() / 255.0
        self.tar_lbl = tar_lbl.cuda()
        self.tar_bbox = tar_bbox.unsqueeze(dim=1).cuda()

    def set_test_input(self, src_img_list, src_lbl_list, src_bbox_list,
                       tar_lbl, tar_bbox,
                       prev_tar_img=None, prev_tar_lbl=None, prev_tar_bbox=None):
        self.src_img_list = [src_img.cuda() / 255.0 for src_img in src_img_list]
        self.src_lbl_list = [src_lbl.cuda() for src_lbl in src_lbl_list]
        self.src_bbox_list = [src_bbox.unsqueeze(dim=1).cuda() for src_bbox in src_bbox_list]
        self.tar_lbl = tar_lbl.cuda()
        self.tar_bbox = tar_bbox.unsqueeze(dim=1).cuda()
        if prev_tar_img is not None:
            self.prev_tar_img = prev_tar_img.cuda() / 255.0
            self.prev_tar_lbl = prev_tar_lbl.cuda()
            self.prev_tar_bbox = prev_tar_bbox.cuda()

    def set_source_num(self, n_source):
        self.n_source = n_source

    def get_grid(self, b, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid

    def forward(self):
        src_img_fea_list = []
        for i in range(self.n_source):
            src_img_fea = self.img_enc(torch.cat([self.src_img_list[i], self.src_lbl_list[i]], dim=1))
            src_img_fea_list.append(src_img_fea)

        tar_lbl_fea = self.lbl_enc(self.tar_lbl)
        b, c, h, w = tar_lbl_fea.size()

        # propagation branch
        tar_lbl_fea_norm = F.normalize(tar_lbl_fea, p=2, dim=1)
        tar_lbl_fea_norm_reshape = tar_lbl_fea_norm.view(b, c, h * w).transpose(1, 2)

        tar_bbox_down = F.interpolate(self.tar_bbox, (h, w), mode="nearest")
        tar_bbox_down_reshape = tar_bbox_down.view(b, 1, h * w).transpose(1, 2)

        pg_rec_tar_img_fea_list = []
        src_img_lbl_fea_norm_list = []
        if self.is_train:
            self.warp_src_img_list = []
            warp_loss_list = []
            ref_mean = self.tar_img.view(b, 3, -1).mean(dim=2).view(b, 3, 1, 1)
            ref_std = self.tar_img.view(b, 3, -1).std(dim=2).view(b, 3, 1, 1)

        for i in range(self.n_source):
            # combine image and label feature together
            # source image feature has already included src_img_fea
            src_img_lbl_fea_norm = F.normalize(src_img_fea_list[i], p=2, dim=1)
            src_img_lbl_fea_norm_list.append(src_img_lbl_fea_norm)
            src_img_lbl_fea_norm_reshape = src_img_lbl_fea_norm.view(b, c, h * w)

            # using bounding box to reduce computational complexity
            src_bbox_down = F.interpolate(self.src_bbox_list[i], (h, w), mode="nearest")
            src_bbox_down_reshape = src_bbox_down.view(b, 1, h * w)

            src_img_lbl_fea_norm_reshape_in_bbox = src_img_lbl_fea_norm_reshape * src_bbox_down_reshape
            tar_lbl_fea_norm_reshape_in_box = tar_lbl_fea_norm_reshape * tar_bbox_down_reshape
            tar_src_feature_mul_in_box = torch.bmm(tar_lbl_fea_norm_reshape_in_box,
                                                   src_img_lbl_fea_norm_reshape_in_bbox)
            src_img_lbl_fea_norm_reshape_out_bbox = src_img_lbl_fea_norm_reshape * (1.0 - src_bbox_down_reshape)
            tar_lbl_fea_norm_reshape_out_box = tar_lbl_fea_norm_reshape * (1.0 - tar_bbox_down_reshape)
            tar_src_feature_mul_out_box = torch.bmm(tar_lbl_fea_norm_reshape_out_box,
                                                    src_img_lbl_fea_norm_reshape_out_bbox)
            tar_src_feature_mul = tar_src_feature_mul_in_box + tar_src_feature_mul_out_box  # [t1s1, t1s2, ..., t1sn; t2s1, ...]
            tar_src_feature_mul = F.softmax(100 * tar_src_feature_mul, dim=2)

            # change to use coordinate translator
            grid2d = self.get_grid(b, h, w, normalize=True).cuda()
            grid2d_reshape = grid2d.view(b, h * w, 2)
            warp_grid2d_reshape = torch.matmul(tar_src_feature_mul, grid2d_reshape)
            warp_grid2d = warp_grid2d_reshape.view(b, h, w, 2)
            pg_rec_tar_img_fea = F.grid_sample(src_img_fea_list[i], warp_grid2d, align_corners=False)
            pg_rec_tar_img_fea_list.append(pg_rec_tar_img_fea)

            if self.is_train:
                _, _, ori_h, ori_w = self.src_img_list[i].size()
                down = ori_h // h
                src_img_down = F.unfold(self.src_img_list[i], down, stride=down)
                src_img_down_reshape = src_img_down.view(b, -1, h, w)
                pg_rec_tar_img_down_reshape = F.grid_sample(src_img_down_reshape, warp_grid2d, align_corners=False)
                pg_rec_tar_img_down = pg_rec_tar_img_down_reshape.view(b, -1, h * w)
                warp_src_img = F.fold(pg_rec_tar_img_down, 256, down, stride=down)
                # renorm images
                gen_mean = warp_src_img.view(b, 3, -1).mean(dim=2).view(b, 3, 1, 1)
                gen_std = warp_src_img.view(b, 3, -1).std(dim=2).view(b, 3, 1, 1)
                norm_warp_src_img = (warp_src_img - gen_mean) / gen_std
                warp_src_img = norm_warp_src_img * ref_std + ref_mean
                if self.use_mask:
                    warp_src_img = warp_src_img * self.fore_mask + self.mask_img * (1 - self.fore_mask)
                self.warp_src_img_list.append(warp_src_img)
                warp_loss = 10 * F.l1_loss(warp_src_img, self.tar_img)
                warp_loss_list.append(warp_loss)

        if self.is_train:
            self.loss_warp = sum(warp_loss_list)

        weighted_pg_rec_tar_img_fea = torch.stack(pg_rec_tar_img_fea_list, dim=1).mean(dim=1)

        # synthesis branch
        sg_rec_tar_img_fea_list = []
        for i in range(self.n_source):
            sg_rec_tar_img_fea = self.fuse_net(src_img_fea_list[i], tar_lbl_fea)
            sg_rec_tar_img_fea_list.append(sg_rec_tar_img_fea)

        weighted_sg_rec_tar_img_fea = torch.stack(sg_rec_tar_img_fea_list, dim=1).mean(dim=1)

        self.rec_tar_img, rec_tar_fea = self.dec(weighted_pg_rec_tar_img_fea, weighted_sg_rec_tar_img_fea)

        if self.use_mask:
            self.rec_tar_img = self.rec_tar_img * self.fore_mask + self.mask_img * (1 - self.fore_mask)

    def paired_backward_D(self, real_img, real_lbl, fake_img):
        fake_st = torch.cat((real_lbl, fake_img), 1)
        pred_fake = self.netD(fake_st.detach())
        loss_D_fake = self.criterionGAN(pred_fake[-1], False)
        real_st = torch.cat((real_lbl, real_img), 1)
        pred_real = self.netD(real_st)
        loss_D_real = self.criterionGAN(pred_real[-1], True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        # add face discriminator
        fake_face_img = self.crop_face(fake_img, real_lbl)
        real_face_img = self.crop_face(real_img, real_lbl)
        pred_fake_face = self.netDF(fake_face_img.detach())
        loss_DF_fake = self.criterionGAN(pred_fake_face[-1], False)
        pred_real_face = self.netDF(real_face_img)
        loss_DF_real = self.criterionGAN(pred_real_face[-1], True)
        loss_DF = (loss_DF_fake + loss_DF_real) * 0.5

        return loss_D, loss_D_fake, loss_D_real, loss_DF, loss_DF_fake, loss_DF_real

    def crop_face(self, image, real_lbl):
        bs, _, h, w = real_lbl.size()
        face_size = image.shape[-2] // 32 * 8
        output = None
        for i in range(bs):
            ys, ye, xs, xe = self.get_face_bbox(real_lbl[i, :, :, :])
            output_i = F.interpolate(image[i:i + 1, :, ys:ye, xs:xe],
                                     size=(face_size, face_size), mode='bilinear',
                                     align_corners=True)
            output = torch.cat([output, output_i]) if i != 0 else output_i
        return output

    def get_face_bbox(self, real_lbl):
        _, h, w = real_lbl.size()
        ylen = xlen = h // 32 * 8
        face_mask = real_lbl[-1, :, :]
        head_mask = real_lbl[1, :, :] + real_lbl[2, :, :] + \
                    real_lbl[3, :, :] + real_lbl[4, :, :]
        face_coord = face_mask.nonzero()
        head_coord = head_mask.nonzero()
        if face_coord.size(0):
            y, x = face_coord[:, 0], face_coord[:, 1]
            ys, ye = y.min().item(), y.max().item()
            xs, xe = x.min().item(), x.max().item()
            xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
            ylen = int((xe - xs) * 2.5)
            ylen = xlen = min(w, max(32, ylen))
            yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
            xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
        elif head_coord.size(0):
            y, x = head_coord[:, 0], head_coord[:, 1]
            ys, ye = y.min().item(), y.max().item()
            xs, xe = x.min().item(), x.max().item()
            xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
            ylen = int((xe - xs) * 2.5)
            ylen = xlen = min(w, max(32, ylen))
            yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
            xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
        else:
            yc = h // 4
            xc = w // 2

        ys, ye = yc - ylen // 2, yc + ylen // 2
        xs, xe = xc - xlen // 2, xc + xlen // 2
        return ys, ye, xs, xe

    def paired_backward_G(self, real_img, real_lbl, fake_img):
        fake_st = torch.cat((real_lbl, fake_img), 1)
        pred_fake = self.netD(fake_st)
        real_st = torch.cat((real_lbl, real_img), 1)
        pred_real = self.netD(real_st)
        loss_G_GAN = self.criterionGAN(pred_fake[-1], True)
        loss_G_FML = 0.0
        for i in range(len(pred_fake) - 1):
            loss_G_FML += self.lambda_FML * self.criterionFML(pred_fake[i], pred_real[i].detach())
        loss_G_VGG = self.lambda_VGG * self.criterionVGG(fake_img, real_img.detach())
        loss_G = loss_G_GAN + loss_G_FML + loss_G_VGG
        # add face discriminator
        fake_face_img = self.crop_face(fake_img, real_lbl)
        real_face_img = self.crop_face(real_img, real_lbl)
        pred_fake_face = self.netDF(fake_face_img)
        pred_real_face = self.netDF(real_face_img)
        loss_GF_GAN = self.criterionGAN(pred_fake_face[-1], True)
        loss_GF_FML = 0.0
        for i in range(len(pred_fake_face) - 1):
            loss_GF_FML += self.lambda_FML * self.criterionFML(pred_fake_face[i], pred_real_face[i].detach())
        loss_GF_VGG = self.lambda_VGG * self.criterionVGG(fake_face_img, real_face_img.detach())
        loss_GF = loss_GF_GAN + loss_GF_FML + loss_GF_VGG
        return loss_G, loss_G_GAN, loss_G_FML, loss_G_VGG, loss_GF, loss_GF_GAN, loss_GF_FML, loss_GF_VGG

    def optimize_parameters(self):
        self.forward()
        # the same video
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netDF, True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.optimizer_DF.zero_grad()
        # tar image
        self.loss_D, self.loss_D_fake, self.loss_D_real, self.loss_DF, self.loss_DF_fake, self.loss_DF_real = \
            self.paired_backward_D(real_img=self.tar_img, real_lbl=self.tar_lbl,
                                   fake_img=self.rec_tar_img)
        (self.loss_D + self.loss_DF).backward()
        self.optimizer_D.step()  # update D's weights
        self.optimizer_DF.step()

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netDF, False)

        # update SG and PG
        self.optimizer_img_enc.zero_grad()  # set G's gradients to zero
        self.optimizer_lbl_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        self.optimizer_fuse_net.zero_grad()
        # tar image
        self.loss_G, self.loss_G_GAN, self.loss_G_FML, self.loss_G_VGG, \
        self.loss_GF, self.loss_GF_GAN, self.loss_GF_FML, self.loss_GF_VGG = \
            self.paired_backward_G(real_img=self.tar_img, real_lbl=self.tar_lbl,
                                   fake_img=self.rec_tar_img)
        self.loss_grad_G = self.lambda_GRAD * self.grad_loss(self.rec_tar_img, self.tar_img)
        (self.loss_G + self.loss_GF + self.loss_grad_G + self.loss_warp).backward()
        self.optimizer_img_enc.step()  # udpate G's weights
        self.optimizer_lbl_enc.step()
        self.optimizer_dec.step()
        self.optimizer_fuse_net.step()

    def grad_loss(self, input, target):
        input_gradx = gradientx(input)
        input_grady = gradienty(input)

        target_gradx = gradientx(target)
        target_grady = gradienty(target)

        return F.l1_loss(torch.abs(target_gradx), torch.abs(input_gradx)) \
               + F.l1_loss(torch.abs(target_grady), torch.abs(input_grady))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_rec_tar_imgs(self, name):
        return getattr(self, name + "_rec_tar_img")

    def print_learning_rate(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        lr_dec = self.optimizers[2].param_groups[0]['lr']
        lr_dis = self.optimizers[-1].param_groups[0]['lr']
        assert lr > 0
        print('lr= %.7f, lr_dec=%.7f, lr_dis=%.7f' % (lr, lr_dec, lr_dis))

    def setup(self, actual_step, batch_size, initial_iter, max_iter, power):
        lr = lr_poly(self.lr, actual_step * batch_size, initial_iter, max_iter, power)
        # Generator
        for optimizer in self.optimizers[:-2]:
            optimizer.param_groups[0]['lr'] = lr
        # Discriminator
        for optimizer in self.optimizers[-2:]:
            optimizer.param_groups[0]['lr'] = 0.5 * lr
        # Decoder
        self.optimizers[2].param_groups[0]['lr'] = self.lambda_dec * lr


def gradientx(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradienty(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def lr_poly(base_lr, iter, initial_iter, max_iter, power):
    return base_lr * ((1 - max(0, float(iter - initial_iter) / (max_iter - initial_iter))) ** (power))


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
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


if __name__ == "__main__":
    import os
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
    label_nc = 2
    bs = 4
    src_img_batch_list = []
    src_lbl_batch_list = []
    src_bbox_batch_list = []
    for i in range(3):
        src_img_batch = torch.rand((bs, 3, 256, 256)).cuda()
        src_lbl_batch = torch.randint(low=0, high=2, size=(bs, label_nc, 256, 256)).cuda().to(torch.float32)
        src_bbox_batch = torch.randint(low=0, high=2, size=(bs, 256, 256)).cuda().to(torch.float32)
        src_img_batch_list.append(src_img_batch)
        src_lbl_batch_list.append(src_lbl_batch)
        src_bbox_batch_list.append(src_bbox_batch)

    tar_img_batch = torch.rand((bs, 3, 256, 256)).cuda()
    tar_lbl_batch = torch.randint(low=0, high=2, size=(bs, label_nc, 256, 256)).cuda().to(torch.float32)
    tar_bbox_batch = torch.randint(low=0, high=2, size=(bs, 256, 256)).cuda().to(torch.float32)

    model = TSNet(is_train=True, label_nc=label_nc,
                  n_blocks=0, debug=False,
                  n_downsampling=3,
                  n_source=3).cuda()
    model.set_train_input(src_img_list=src_img_batch_list,
                          src_lbl_list=src_lbl_batch_list,
                          src_bbox_list=src_bbox_batch_list,
                          tar_img=tar_img_batch, tar_lbl=tar_lbl_batch,
                          tar_bbox=tar_bbox_batch)
    model.optimize_parameters()
