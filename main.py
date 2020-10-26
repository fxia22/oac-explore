import os
import os.path as osp

import argparse
import torch

import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer, gibson_env_producer, parallel_gibson_env_producer, gibson_stadium_env_producer, parallel_gibson_stadium_env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic, ReLMoGenCritic, ReLMoGenTanhGaussianPolicy
from trainer.trainer import SACTrainer
from networks import FlattenMlp
from rl_algorithm import BatchRLAlgorithm

import ray
import logging
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.WARNING,

    # The amount of memory (in bytes)
    object_store_memory=1073741824, # 1g
    redis_max_memory=1073741824 # 1g
)


def get_current_branch(dir):

    from git import Repo

    repo = Repo(dir)
    return repo.active_branch.name


def get_policy_producer(observation_space, action_dim, custom_initialization):

    def policy_producer(deterministic=False):

        policy = ReLMoGenTanhGaussianPolicy(
            observation_space=observation_space, action_dim=action_dim, custom_initialization=custom_initialization,
        )

        if deterministic:
            policy = MakeDeterministic(policy)

        return policy

    return policy_producer


def get_q_producer(observation_space, action_dim, custom_initialization):
    def q_producer():
        return ReLMoGenCritic(observation_space=observation_space, action_dim=action_dim, custom_initialization=custom_initialization)

    return q_producer


def experiment(variant, prev_exp_state=None):

    domain = variant['domain']
    seed = variant['seed']
    num_parallel = variant['num_parallel']
    custom_initialization = variant['custom_initialization']

    expl_env = parallel_gibson_env_producer(num_env=num_parallel)
    #expl_env = parallel_gibson_stadium_env_producer(num_env=num_parallel)

    #obs_dim = expl_env.observation_space.low.size
    observation_space = expl_env.observation_space
    action_dim = expl_env.action_space.low.size

    # Get producer function for policy and value functions
    q_producer = get_q_producer(observation_space, action_dim, custom_initialization)
    policy_producer = get_policy_producer(
        observation_space, action_dim, custom_initialization)
    # Finished getting producer

    remote_eval_path_collector = RemoteMdpPathCollector.remote(
        domain, seed * 10 + 1,
        policy_producer,
        max_num_epoch_paths_saved=1
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        max_num_epoch_paths_saved=1,
    )
    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )
    trainer = SACTrainer(
        policy_producer,
        q_producer,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )

    algorithm = BatchRLAlgorithm(
        trainer=trainer,

        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,

        replay_buffer=replay_buffer,
        optimistic_exp_hp=variant['optimistic_exp'],
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    if prev_exp_state is not None:

        expl_path_collector.restore_from_snapshot(
            prev_exp_state['exploration'])

        ray.get([remote_eval_path_collector.restore_from_snapshot.remote(
            prev_exp_state['evaluation_remote'])])
        ray.get([remote_eval_path_collector.set_global_pkg_rng_state.remote(
            prev_exp_state['evaluation_remote_rng_state']
        )])

        replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])

        trainer.restore_from_snapshot(prev_exp_state['trainer'])

        set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='invertedpendulum')
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.0)

    # Training param
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=1)
    parser.add_argument('--num_parallel', type=int, default=2)
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=1000)

    parser.add_argument('--dir_suffix', type=str, default="exp")
    parser.add_argument('--custom_initialization', dest='custom_initialization', action='store_true')

    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    log_dir = osp.join(
        get_current_branch('./'),

        # Algo kwargs portion
        f'num_expl_steps_per_train_loop_{args.num_expl_steps_per_train_loop}_num_trains_per_train_loop_{args.num_trains_per_train_loop}_{args.dir_suffix}'

        # optimistic exploration dependent portion
        f'beta_UB_{args.beta_UB}_delta_{args.delta}',
    )

    if should_include_domain:
        log_dir = osp.join(log_dir, args.domain)

    if should_include_seed:
        log_dir = osp.join(log_dir, f'seed_{args.seed}')

    if should_include_base_log_dir:
        log_dir = osp.join(args.base_log_dir, log_dir)

    return log_dir


if __name__ == "__main__":

    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        custom_initialization=None,
        replay_buffer_size=int(1E5),
        num_parallel=None,
        algorithm_kwargs=dict(
            num_epochs=50000,
            num_train_loops_per_epoch=None,
            num_eval_steps_per_epoch=None,
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            max_path_length=None,
            min_num_steps_before_training=int(1E4),
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        optimistic_exp={}
    )

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)
    variant['seed'] = args.seed
    variant['domain'] = args.domain
    variant['num_parallel'] = args.num_parallel
    variant['custom_initialization'] = args.custom_initialization
    variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_train_loops_per_epoch
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_parallel * args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['max_path_length'] = args.num_parallel * args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = args.num_parallel * args.num_eval_steps_per_epoch

    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0
    variant['optimistic_exp']['beta_UB'] = args.beta_UB
    variant['optimistic_exp']['delta'] = args.delta

    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None

    run_experiment_here(experiment, variant,
                        seed=args.seed,
                        use_gpu=not args.no_gpu and torch.cuda.is_available(),
                        gpu_id=gpu_id,

                        # Save the params every snapshot_gap and override previously saved result
                        snapshot_gap=50000,
                        snapshot_mode='last_every_gap',

                        log_dir=variant['log_dir']

                        )
