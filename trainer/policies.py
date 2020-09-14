import numpy as np
import torch
from torch import nn as nn

from utils.core import eval_np
from networks import Mlp


import torch
from torch.distributions import Distribution, Normal
import utils.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class TanhGaussianPolicy(Mlp):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, std,
            pre_tanh_value,
        )

    def reset(self):
        pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self,
                 observation_space,
                 single_branch_size=256,
                 ):
        super().__init__()

        if "sensor" in observation_space.spaces:
            self._n_non_vis_sensor = observation_space.spaces["sensor"].shape[0]
        else:
            self._n_non_vis_sensor = 0

        self._single_branch_size = single_branch_size
        self.hidden_size = 0

        self.feature_linear = nn.Sequential(
            nn.Linear(self._n_non_vis_sensor,
                        self._single_branch_size),
            nn.ReLU()
        )
        self.hidden_size += single_branch_size

        self._cnn_layers_params = [
                (32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        self.cnn = self._init_perception_model(observation_space)
        if not self.is_blind:
            self.hidden_size += single_branch_size

        self._cnn_1d_layers_params = [
            (32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        self.cnn_1d = self._init_lidar_model(observation_space)
        if self._n_input_scan > 0:
            self.hidden_size += single_branch_size

        #self.layer_init()
        self.train()

    def _init_lidar_model(self, observation_space):
        if "scan" in observation_space.spaces:
            self._n_input_scan = observation_space.spaces["scan"].shape[1]
        else:
            self._n_input_scan = 0

        if self._n_input_scan == 0:
            return nn.Sequential()

        cnn_dim = observation_space.spaces["scan"].shape[0]
        for _, kernel_size, stride, padding in self._cnn_1d_layers_params:
            cnn_dim = self._conv_1d_output_dim(
                dimension=cnn_dim,
                padding=padding,
                dilation=1,
                kernel_size=kernel_size,
                stride=stride,
            )

        cnn_layers = []
        prev_out_channels = None
        for i, (out_channels, kernel_size, stride, padding) in enumerate(self._cnn_1d_layers_params):
            if i == 0:
                in_channels = self._n_input_scan
            else:
                in_channels = prev_out_channels
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ))
            if i != len(self._cnn_1d_layers_params) - 1:
                cnn_layers.append(nn.ReLU())
            prev_out_channels = out_channels

        cnn_layers += [
            Flatten(),
            nn.Linear(self._cnn_1d_layers_params[-1][0] * cnn_dim,
                      self._single_branch_size),
            nn.ReLU(),
        ]
        return nn.Sequential(*cnn_layers)

    def _init_perception_model(self, observation_space):
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32)

        if self.is_blind:
            return nn.Sequential()
        else:
            for _, kernel_size, stride, padding in self._cnn_layers_params:
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([padding, padding], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(
                        [kernel_size, kernel_size], dtype=np.float32),
                    stride=np.array([stride, stride], dtype=np.float32),
                )

            cnn_layers = []
            prev_out_channels = None
            for i, (out_channels, kernel_size, stride, padding) in enumerate(self._cnn_layers_params):
                if i == 0:
                    in_channels = self._n_input_rgb + self._n_input_depth
                else:
                    in_channels = prev_out_channels
                cnn_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ))
                if i != len(self._cnn_layers_params) - 1:
                    cnn_layers.append(nn.ReLU())
                prev_out_channels = out_channels

            cnn_layers += [
                Flatten(),
                nn.Linear(self._cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                          self._single_branch_size),
                nn.ReLU(),
            ]
            return nn.Sequential(*cnn_layers)

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i]
                      * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _conv_1d_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        out_dimension = int(np.floor(
            ((dimension + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1))
        return out_dimension

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, val=0)

        for layer in self.cnn_1d:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward_perception_model(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)

    def forward_lidar_model(self, observations):
        lidar_input = observations["scan"]
        # permute tensor to dimension [BATCH x CHANNEL x WIDTH]
        lidar_input = lidar_input.permute(0, 2, 1)
        return self.cnn_1d(lidar_input)

    def forward(self, observations):
        x = self.feature_linear(observations["sensor"])
        if not self.is_blind:
            perception_embed = self.forward_perception_model(observations)
            if x is None:
                x = perception_embed
            else:
                x = torch.cat([x, perception_embed], dim=1)

        if self._n_input_scan > 0:
            lidar_embed = self.forward_lidar_model(observations)
            if x is None:
                x = lidar_embed
            else:
                x = torch.cat([x, lidar_embed], dim=1)

        return x


class ReLMoGenTanhGaussianPolicy(nn.Module):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            observation_space,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(observation_space)
        self.log_std = None
        self.std = std

        self.fc = nn.Linear(self.encoder.hidden_size, 256)
        self.relu = nn.ReLU()

        self.last_fc = nn.Linear(256, action_dim)
        if std is None:
            self.last_fc_log_std = nn.Linear(256, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions({k: v[None] for k,v in obs_np.items()}, deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = self.encoder(obs)
        h = self.relu(self.fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, std,
            pre_tanh_value,
        )

    def reset(self):
        pass


class ReLMoGenCritic(nn.Module):
    def __init__(
            self,
            observation_space,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__()

        self.encoder = Encoder(observation_space)
        self.obs_fc = nn.Linear(self.encoder.hidden_size, 256)
        self.obs_relu = nn.ReLU()
        self.action_fc = nn.Linear(action_dim, 256)
        self.action_relu = nn.ReLU()
        self.joint_fc = nn.Linear(512, 256)
        self.joint_relu = nn.ReLU()
        self.last_fc = nn.Linear(256, 1)

        """
        self.fc1 = nn.Linear(14, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.last_fc = nn.Linear(256, 1)
        """

    def forward(self, obs, action):

        h1 = self.encoder(obs)
        h1 = self.obs_relu(self.obs_fc(h1))
        h2 = self.action_relu(self.action_fc(action))
        h = self.joint_relu(self.joint_fc(torch.cat((h1, h2), dim=1)))
        q = self.last_fc(h)

        """
        h = self.relu1(self.fc1(obs['sensor']))
        h = self.relu2(self.fc2(h))
        q = self.last_fc(h)
        """

        return q

    def reset(self):
        pass

class MakeDeterministic(object):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def reset(self):
        pass


def policy_producer(obs_dim, action_dim, hidden_sizes, deterministic=False):

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    if deterministic:
        policy = MakeDeterministic(policy)

    return policy

if __name__ == "__main__":
    from collections import OrderedDict
    import gym
    observation_space = OrderedDict()
    observation_space['sensor'] = gym.spaces.Box(low=-np.inf,
                            high=np.inf,
                            shape=(55,),
                            dtype=np.float32)
    observation_space['rgb'] = gym.spaces.Box(low=-np.inf,
                        high=np.inf,
                        shape=(128, 128, 3),
                        dtype=np.float32)
    observation_space['depth'] = gym.spaces.Box(low=-np.inf,
                    high=np.inf,
                    shape=(128, 128, 1),
                    dtype=np.float32)
    observation_space['scan'] = gym.spaces.Box(low=-np.inf,
                high=np.inf,
                shape=(220, 1),
                dtype=np.float32)

    observation_space = gym.spaces.Dict(observation_space)
    actor = ReLMoGenTanhGaussianPolicy(observation_space, 3)
    critic = ReLMoGenCritic(observation_space, 3)
    obs = {
        'sensor': torch.from_numpy(np.zeros((8, 55), dtype=np.float32)),
        'rgb': torch.from_numpy(np.zeros((8, 128, 128, 3), dtype=np.float32)),
        'depth': torch.from_numpy(np.zeros((8, 128, 128, 1), dtype=np.float32)),
        'scan': torch.from_numpy(np.zeros((8, 220, 1), dtype=np.float32))
    }
    action = torch.from_numpy(np.zeros((8, 3), dtype=np.float32))
    feature = actor(obs)
    print(feature)
    q = critic(obs, action)
    print(q.shape)
