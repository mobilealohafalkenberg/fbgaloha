import argparse
from copy import deepcopy
from itertools import repeat
import os
import pickle

from constants import TASK_CONFIGS
import numpy as np
import torch
from tqdm import tqdm
import wandb

from policy import (
    ACTPolicy,
    CNNMLPPolicy,
    DiffusionPolicy
)
from utils import (
    load_data,
    compute_dict_mean,
    set_seed,
)


def main(args):
    set_seed(args['seed'])

    # Command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # Get task parameters
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # Fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'action_dim': 16,
            'backbone': backbone,
            'camera_names': camera_names,
            'dec_layers': dec_layers,
            'dim_feedforward': args['dim_feedforward'],
            'enc_layers': enc_layers,
            'hidden_dim': args['hidden_dim'],
            'kl_weight': args['kl_weight'],
            'lr_backbone': lr_backbone,
            'lr': args['lr'],
            'nheads': nheads,
            'no_encoder': args['no_encoder'],
            'num_queries': args['chunk_size'],
            'vq_class': args['vq_class'],
            'vq_dim': args['vq_dim'],
            'vq': args['use_vq'],
        }
    elif policy_class == 'Diffusion':
        policy_config = {
            'action_dim': 16,
            'action_horizon': 8,
            'camera_names': camera_names,
            'ema_power': 0.75,
            'lr': args['lr'],
            'num_inference_timesteps': 10,
            'num_queries': args['chunk_size'],
            'observation_horizon': 1,
            'prediction_horizon': args['chunk_size'],
            'vq': False,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'backbone': backbone,
            'camera_names': camera_names,
            'lr_backbone': lr_backbone,
            'lr': args['lr'],
            'num_queries': 1,
        }
    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'actuator_config': actuator_config,
        'camera_names': camera_names,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'lr': args['lr'],
        'num_steps': num_steps,
        'onscreen_render': onscreen_render,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'real_robot': False,  # Set to False for training without the robot
        'resume_ckpt_path': resume_ckpt_path,
        'save_every': save_every,
        'seed': args['seed'],
        'state_dim': state_dim,
        'task_name': task_name,
        'temporal_agg': args['temporal_agg'],
        'validate_every': validate_every,
    }

    # Create checkpoint directory if it doesn't exist
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]

    # Initialize Weights & Biases for logging
    wandb.init(
        project="ExampleProject",
        reinit=True,
        entity="ExampleEntity",
        name=expr_name,
    )
    wandb.config.update(config)

    # Save configuration
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    # Load training and validation data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        batch_size_train,
        batch_size_val,
        args['chunk_size'],
        args['skip_mirrored_data'],
        policy_class,
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
    )

    # Save dataset statistics
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Train the policy
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # Save the best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best checkpoint saved with validation loss {min_val_loss:.6f} at step {best_step}')
    wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class in ['ACT', 'CNNMLP', 'Diffusion']:
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data

    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()

    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resumed policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):
        # Validation step
        if step % validate_every == 0:
            print('Validating...')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
            wandb.log(validation_summary, step=step)
            print(f'Validation loss: {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # Training step
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # Backward pass and optimization
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step)

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    # Save the final model
    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training completed:\nSeed {seed}, best validation loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} completed')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--onscreen_render',
        action='store_true',
        help='Render training onscreen',
    )
    parser.add_argument(
        '--ckpt_dir',
        action='store',
        type=str,
        help='Checkpoint directory',
        required=True,
    )
    parser.add_argument(
        '--policy_class',
        action='store',
        type=str,
        default='ACT',
        help='The desired policy class',
        choices=['ACT', 'Diffusion', 'CNNMLP'],
    )
    parser.add_argument(
        '--task_name',
        action='store',
        type=str,
        help='Name of the task. Must be in task configurations',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        action='store',
        type=int,
        help='Training batch size',
        required=True
    )
    parser.add_argument(
        '--seed',
        action='store',
        type=int,
        help='Training seed',
        required=True
    )
    parser.add_argument(
        '--num_steps',
        action='store',
        type=int,
        help='Number of training steps',
        required=True
    )
    parser.add_argument(
        '--lr',
        action='store',
        type=float,
        help='Training learning rate',
        required=True
    )
    parser.add_argument(
        '--validate_every',
        action='store',
        type=int,
        default=500,
        help='Number of steps between validations during training',
        required=False,
    )
    parser.add_argument(
        '--save_every',
        action='store',
        type=int,
        default=500,
        help='Number of steps between checkpoints during training',
        required=False,
    )
    parser.add_argument(
        '--resume_ckpt_path',
        action='store',
        type=str,
        help='Path to checkpoint to resume training from',
        required=False,
    )
    parser.add_argument(
        '--skip_mirrored_data',
        action='store_true',
        help='Skip mirrored data during training',
        required=False,
    )
    parser.add_argument(
        '--actuator_network_dir',
        action='store',
        type=str,
        help='Actuator network directory',
        required=False,
    )
    parser.add_argument(
        '--history_len',
        action='store',
        type=int,
    )
    parser.add_argument(
        '--future_len',
        action='store',
        type=int,
    )
    parser.add_argument(
        '--prediction_len',
        action='store',
        type=int,
    )

    # Arguments for ACT policy
    parser.add_argument(
        '--kl_weight',
        action='store',
        type=int,
        help='KL Weight',
        required=False,
    )
    parser.add_argument(
        '--chunk_size',
        action='store',
        type=int,
        help='Chunk size',
        required=False,
    )
    parser.add_argument(
        '--hidden_dim',
        action='store',
        type=int,
        help='Hidden dimension size',
        required=False,
    )
    parser.add_argument(
        '--dim_feedforward',
        action='store',
        type=int,
        help='Feedforward dimension size',
        required=False,
    )
    parser.add_argument(
        '--temporal_agg',
        action='store_true',
        help='Use temporal aggregation',
    )
    parser.add_argument(
        '--use_vq',
        action='store_true',
        help='Use vector quantization',
    )
    parser.add_argument(
        '--vq_class',
        action='store',
        type=int,
        help='Number of VQ classes',
    )
    parser.add_argument(
        '--vq_dim',
        action='store',
        type=int,
        help='Dimension of VQ embeddings',
    )
    parser.add_argument(
        '--no_encoder',
        action='store_true',
        help='Do not use an encoder in the policy',
    )

    main(vars(parser.parse_args()))
