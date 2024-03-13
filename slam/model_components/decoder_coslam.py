# Package imports
import tinycudann as tcnn
import torch
import torch.nn as nn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4):
        super(ColorNet, self).__init__()
        self.input_ch = input_ch
        self.geo_feat_dim = config.geo_feat_dim
        self.hidden_dim_color = config.hidden_dim_color
        self.num_layers_color = config.num_layers_color
        self.tcnn_network = config.tcnn_network

        self.model = self.get_model(self.tcnn_network)

    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': 'ReLU',
                    'output_activation': 'None',
                    'n_neurons': self.hidden_dim_color,
                    'n_hidden_layers': self.num_layers_color - 1,
                },
                # dtype=torch.float
            )

        color_net = []
        for layer in range(self.num_layers_color):
            if layer == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            if layer == self.num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if layer != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))


class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3):
        super(SDFNet, self).__init__()
        self.input_ch = input_ch
        self.geo_feat_dim = config.geo_feat_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.tcnn_network = config.tcnn_network

        self.model = self.get_model(tcnn_network=self.tcnn_network)

    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': 'ReLU',
                    'output_activation': 'None',
                    'n_neurons': self.hidden_dim,
                    'n_hidden_layers': self.num_layers - 1,
                },
                # dtype=torch.float
            )
        else:
            sdf_net = []
            for layer in range(self.num_layers):
                if layer == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim

                if layer == self.num_layers - 1:
                    # 1 sigma + 15 SH features for color
                    out_dim = 1 + self.geo_feat_dim
                else:
                    out_dim = self.hidden_dim

                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if layer != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))


class ColorSDFNet(nn.Module):
    """Color grid + SDF grid."""
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet, self).__init__()
        self.color_net = ColorNet(config, input_ch=input_ch + input_ch_pos)
        self.sdf_net = SDFNet(config, input_ch=input_ch + input_ch_pos)

    def forward(self, embed, embed_pos, embed_color):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1),
                             return_geo=True)
        else:
            h = self.sdf_net(embed, return_geo=True)

        sdf, geo_feat = h[..., :1], h[..., 1:]
        if embed_pos is not None:
            rgb = self.color_net(
                torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))

        return torch.cat([rgb, sdf], -1)


class ColorSDFNet_v2(nn.Module):
    """No color grid."""
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v2, self).__init__()
        self.color_net = ColorNet(config, input_ch=input_ch_pos)
        self.sdf_net = SDFNet(
            config,
            input_ch=input_ch + input_ch_pos,
        )

    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1),
                             return_geo=True)
        else:
            h = self.sdf_net(embed, return_geo=True)

        sdf, geo_feat = h[..., :1], h[..., 1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))

        return torch.cat([rgb, sdf], -1)
