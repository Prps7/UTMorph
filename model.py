from networks import *

class UTMorph(nn.Module):
    def __init__(self, base_chan, reduce_size=8, block_list='234', num_blocks=[1, 2, 4],
                 projection='interp', num_heads=[2, 4, 8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True,
                 rel_pos=True, feature=True, proj_att=True):
        super().__init__()
        self.patchemb = patchembed()
        self.feature = feature
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet((128,128),
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)
        self.inc = []

        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                            attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                            projection=projection, rel_pos=rel_pos, proj_att=proj_att))
            self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4],
                                      dim_head=base_chan // num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop,
                                      reduce_size=reduce_size, projection=projection, rel_pos=rel_pos, proj_att=proj_att)

        else:
            self.inc.append(Conv2dReLU(32, 16, 3, 1, use_batchnorm=False))
            self.up4 = DecoderBlock(base_chan // 2, 32, skip_channels=16,
                                    use_batchnorm=False)  # 384, 80, 80, 128
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1 = down_block_trans(32, 48, num_block=num_blocks[-4], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-4], dim_head=2 * base_chan // num_heads[-4],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos, proj_att=proj_att)
            self.up3 = up_block_trans(base_chan, base_chan // 2, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos, proj_att=proj_att)
        else:
            self.down1 = Conv2dReLU(32, 48, 3, 1, use_batchnorm=False)
            self.up3 = DecoderBlock(base_chan, base_chan // 2, skip_channels=base_chan // 2,
                                    use_batchnorm=False)

        if '2' in block_list:
            self.down2 = down_block_trans(48, base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-3], dim_head=4 * base_chan // num_heads[-3],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos, proj_att=proj_att)
            self.up2 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos, proj_att=proj_att)

        else:
            self.down2 = down_block(48, base_chan, (2, 2), num_block=2)
            self.up2 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-2], dim_head=8 * base_chan // num_heads[-2],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos, proj_att=proj_att)
            self.up1 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-1], dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos, proj_att=proj_att)

        else:
            self.down3 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
            self.up1 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-1],
                                          bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                          dim_head=16 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                          proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                          rel_pos=rel_pos, proj_att=proj_att)
        else:
            self.down4 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)

        self.reg_head = RegistrationHead(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
        )
        self.integrate = VecInt((128,128), 7)

    def forward(self, source, target):
        if self.feature:
            source_feat = self.feature_extractor(source)
            target_feat = self.feature_extractor(target)
        else:
            source_feat = source.repeat(1,16,1,1)
            target_feat = target.repeat(1,16,1,1)
        x = torch.cat([source_feat, target_feat], dim=1)

        x1 = self.inc(x)
        x_down = self.avg_pool(x)
        x2 = self.down1(x_down)
        # x3 = self.down2(x2)
        x3 = self.patchemb(x)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)

        out = self.up4(out, x1)

        flow = self.reg_head(out)
        pos_flow = self.integrate(flow)

        return pos_flow


if __name__ == "__main__":

    img_size = (2, 1, 128, 128)
    fix_img = np.random.rand(*img_size)
    mov_img = np.random.rand(*img_size)
    fix_img_tensor = torch.from_numpy(fix_img).float().to("cuda")
    mov_img_tensor = torch.from_numpy(mov_img).float().to("cuda")
    model = UTMorph(96, reduce_size=8, block_list="234", num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1,
                  proj_drop=0.1, rel_pos=True, maxpool=True, feature=True, proj_att=True).to("cuda")

    print(model)
    disp = model(fix_img_tensor,mov_img_tensor)
    print(disp.shape)
