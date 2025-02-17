import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class Period_encoder_decoder_predication(nn.Module):
    def __init__(self, seq_in_len, seq_out_len, enc_in, period_len):
        super(Period_encoder_decoder_predication, self).__init__()
        self.seq_len = seq_in_len
        self.pred_len = seq_out_len
        self.enc_in = enc_in
        self.period_len = period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear1 = nn.Linear(self.seg_num_x, self.seg_num_y)
        self.linear2 = nn.Linear(self.period_len, self.period_len)

    def forward(self, x):
        #  b, c, s
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.linear1(x).permute(0, 2, 1)
        y = self.linear2(y).permute(0, 2, 1)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        #  b, c, s
        return y


class Multi_period_predication(nn.Module):
    def __init__(self, seq_in_len, seq_out_len, enc_in, period_channels, device='cuda'):
        super(Multi_period_predication, self).__init__()
        self.seq_len = seq_in_len
        self.pred_len = seq_out_len
        self.enc_in = enc_in
        self.period_channels = period_channels
        self.device = device
        self.period_encoders = nn.ModuleList(
            [
                Period_encoder_decoder_predication(
                    seq_in_len, seq_out_len, enc_in, period_len.item() if isinstance(
                        period_len, torch.Tensor) else period_len)
                for period_len in period_channels
            ])
        self.period_weights = nn.Parameter(torch.ones(len(period_channels), 1, 1).to(device))

    def forward(self, x):
        # x: [batch_size, enc_in, seq_len]
        batch_size = x.shape[0]
        period_outputs = []
        for i, period_encoder in enumerate(self.period_encoders):
            period_outputs.append(period_encoder(x))
        period_outputs = torch.stack(period_outputs, dim=1)  # [batch_size, num_periods, enc_in, pred_len]
        weighted_output = (period_outputs * self.period_weights).sum(dim=1)  # [batch_size, enc_in, pred_len]
        return weighted_output

class Sequential_projection(nn.Module):
    def __init__(self, seq_in_len, seq_out_len):
        super(Sequential_projection, self).__init__()
        self.seq_len = seq_in_len
        self.pred_len = seq_out_len
        # self.kan = KAN([self.seq_len, self.pred_len], grid_size=5, base_activation=nn.Identity)
        self.linear = nn.Linear(self.seq_len, self.pred_len, bias=False)

    def forward(self, x):
        #  b, n, s
        # y = self.kan(x)
        y = self.linear(x)
        #  b, n, s
        return y


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        if self.hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")
        layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            # KAN([input_size + hidden_size,self.hparams["backbone_units"]], grid_size=5, base_activation=nn.Identity),
            backbone_activation(),
        ]
        for i in range(1, self.hparams["backbone_layers"]):
            layer_list.append(
                # KAN([self.hparams["backbone_units"], self.hparams["backbone_units"]],grid_size=5, base_activation=nn.Identity)
                nn.Linear(
                    self.hparams["backbone_units"], self.hparams["backbone_units"]
                )
            )
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys():
                layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
            #exp(ts-ts*ln(ts)) = exp(ts * (1 - ln(ts) ) ) 分子
            # ts 分母
            # self.attn_num = self.hparams["backbone_units"]
            #
            # self.query = torch.nn.Linear(self.attn_num, self.attn_num)
            # self.key = torch.nn.Linear(self.attn_num, self.attn_num)
            # self.value = torch.nn.Linear(self.attn_num, self.attn_num)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):

        batch_size = input.size(0)
        ts = ts.view(batch_size, 1).unsqueeze(1)
        w_ts = torch.exp(ts * (1 - 2 * torch.log(ts)))
        x = torch.cat([input, hx], 2)
        x = self.backbone(x)
        if self._minimal:
            # Solution
            ff1 = self.ff1(x)
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            ff1 = self.tanh(self.ff1(x))  # g
            ff2 = self.tanh(self.ff2(x))  # h
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * w_ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden




class Cfc(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature,
        hparams,
        return_sequences=False,
        use_mixed=False,
        use_ltc=False,
    ):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences
        self.in_lens = hparams['in_lens']
        self.out_lens = hparams['out_lens']
        self.period_lens = torch.tensor(hparams['period_len'], dtype=torch.int32)

        if use_ltc:
            self.rnn_cell = LTCCell(in_features, hidden_size)

        else:
            self.rnn_cell_forward = CfcCell(in_features, hidden_size, hparams)
        self.use_mixed = use_mixed
        self.fc = nn.Linear(self.hidden_size, self.out_feature)
        self.encoder = Multi_period_predication(self.in_lens, self.in_lens, self.in_features, self.period_lens)
        self.x_pro = Multi_period_predication(self.in_lens, self.in_lens, self.in_features, self.period_lens)
        self.Sequential_projection = Sequential_projection(self.in_lens * 2, self.out_lens)

#2024年11月1日10点07分
#   KANConv
#   paralle CFC
#   预分解 不可取
    def forward(self, x, timespans=None, mask=None):
        # 预处理
        x = x.squeeze(1).permute(0, 2, 1)  # b t n

        # 分布正变换
        seq_mean = torch.mean(x, dim=1).unsqueeze(2).permute(0, 2, 1)  # b, 1, n
        x = x - seq_mean  # b t n
        seq_mean_out = seq_mean[:, :, 0:3]
        # 交叉编码投影
        x_php = self.encoder(x).permute(0, 2, 1)  # b, t, n
        x = x_php
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)
        module_len = 24
        module_num = 7
        x = x.permute(0, 2, 1).reshape(batch_size, true_in_features, module_num, module_len).permute(0, 3, 2, 1)
        timespans_forward = torch.ones((batch_size, module_len)).to(device)
        h_state_forward = torch.zeros((batch_size, module_num, self.hidden_size), device=device)
        output_sequence_forward = []
        for t in range(module_len):
            inputs_forward = x[:, t]
            ts_forward = timespans_forward[:, t].squeeze()
            h_state_forward = self.rnn_cell_forward.forward(inputs_forward, h_state_forward, ts_forward)
            if self.return_sequences:
                output_sequence_forward.append(h_state_forward)
        if self.return_sequences:
            readout = torch.stack(output_sequence_forward, dim=1)
        else:
            readout = self.fc(h_state_forward)

        timespans_forward_new = torch.zeros((batch_size, module_len)).to(device)#.cumsum(dim=1)
        timespans_forward_new += (1 / 168)  # 168 24 12
        h_state_forward_new = torch.zeros((batch_size, module_num, self.hidden_size), device=device)
        output_sequence_forward_new = []

        x_pro = self.x_pro(x_php).permute(0, 2, 1)
        x_pro = x_pro.permute(0, 2, 1).reshape(batch_size, true_in_features, 7, module_len).permute(0, 3, 2, 1)
        for t in range(module_len):
            inputs_forward_new = x_pro[:, t]
            ts_forward_new = timespans_forward_new[:, t].squeeze()
            h_state_forward_new = self.rnn_cell_forward.forward(inputs_forward_new, h_state_forward_new, ts_forward_new)
            if self.return_sequences:
                output_sequence_forward_new.append(h_state_forward_new)
        if self.return_sequences:
            readout_new = torch.stack(output_sequence_forward_new, dim=1)
        readout = torch.cat((readout, readout_new), dim=2)
        readout = readout.permute(0, 2, 1, 3).reshape(batch_size, module_num*module_len*2, self.hidden_size)
        readout = self.Sequential_projection(readout.permute(0, 2, 1)).permute(0, 2, 1)
        readout = self.fc(readout)
        # 分布逆变换
        readout = readout + seq_mean_out #+ x_php_res #+ x_pdp.permute(0, 2, 1)
        return readout


class LTCCell(nn.Module):
    def __init__(
        self,
        in_features,
        units,
        ode_unfolds=2,
        epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # self.softplus = nn.Softplus()
        self.softplus = nn.Identity()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )


    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / (
            (elapsed_time + 1) / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self.softplus(self._params["w"]) * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self.softplus(self._params["gleak"]) * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
            if torch.any(torch.isnan(v_pre)):
                breakpoint()
        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts):
        # Regularly sampled mode (elapsed time = 1 second)
        ts = ts.view((-1, 1))
        inputs = self._map_inputs(input)

        next_state = self._ode_solver(inputs, hx, ts)

        # outputs = self._map_outputs(next_state)

        return next_state



