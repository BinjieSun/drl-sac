#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from logger import Logger
from replay_buffer import ReplayBuffer
import utils

from torch.utils.tensorboard import SummaryWriter  

import gymnasium as gym
import hydra
from hydra.utils import get_original_cwd

def save_summary(writer, global_step, tag, avg_reward):
    # for k, v in lr_dict.items():
    #     writer.add_scalar(f'{tag}/{k}', v, global_step)
    if avg_reward is not None:
        writer.add_scalar(f'{tag}/average return', avg_reward, global_step)
    writer.flush()

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        project_root = get_original_cwd()
        self.ckpt_dir = os.path.join(project_root, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        saved_tensorboard_path = os.path.join(project_root, 'summary')
        os.makedirs(saved_tensorboard_path, exist_ok=True)
        self.tb_writer = SummaryWriter(saved_tensorboard_path)

        print(f'Checkpoint directory: {self.ckpt_dir}')
        
        # set the frquency to save checkpoint
        self.ckpt_frequency = cfg.get('ckpt_frequency', 100000)

        print(f"Config ckpt_frequency: {cfg.ckpt_frequency}, type: {type(cfg.ckpt_frequency)}")

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # Create training environment
        self.env = utils.make_env(cfg)

        # Create a separate evaluation environment with video recording capabilities
        if cfg.env_type == 'dmc':
            self.eval_env = utils.make_dmc_env(cfg, for_evaluation=True)
        else:
            self.eval_env = utils.make_gym_env(cfg, for_evaluation=True)

        # Handle different observation space types
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            # For DMC environments with Dict observation space
            # Flatten the observation space to make it compatible with your agent
            self.env = gym.wrappers.FlattenObservation(self.env)
            self.eval_env = gym.wrappers.FlattenObservation(self.eval_env)
        
        # Now we can safely get the observation dimension
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]

        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        
        # Create video directory
        self.video_dir = os.path.join(self.work_dir, 'video') if cfg.save_video else None
        if self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

        self.step = 0

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f'checkpoint_{self.step}.pt')
        
        # build checkpoint dict
        ckpt = {
            'step': self.step,
            'agent': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            'replay_buffer': self.replay_buffer.state_dict() if hasattr(self.replay_buffer, 'state_dict') else None,
            'optimizer': self.agent.optimizer.state_dict() if hasattr(self.agent, 'optimizer') else None
        }
        
        # save checkpoint
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path} at step {self.step}")
        
        # save latest checkpoint
        latest_path = os.path.join(self.ckpt_dir, 'latest.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(ckpt_path), latest_path)

    def load_checkpoint(self):
        latest_path = os.path.join(self.ckpt_dir, 'latest.pt')
        
        if os.path.exists(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location=self.device)
                
                # recover
                self.step = ckpt['step']
                
                if ckpt['agent'] is not None and hasattr(self.agent, 'load_state_dict'):
                    self.agent.load_state_dict(ckpt['agent'])
                
                if ckpt['replay_buffer'] is not None and hasattr(self.replay_buffer, 'load_state_dict'):
                    self.replay_buffer.load_state_dict(ckpt['replay_buffer'])
                
                if ckpt['optimizer'] is not None and hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(ckpt['optimizer'])
                
                print(f"Resumed training from step {self.step}")
                return True
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                return False
        else:
            print("No checkpoint found. Starting training from scratch.")
            return False

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs, info = self.eval_env.reset()
            self.agent.reset()
            terminated, truncated = False, False
            episode_reward = 0
            while not (terminated or truncated): 
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward

            # Save video for the first evaluation episode
            if episode == 0 and self.cfg.save_video:
                try:
                    # Check if this env has frames to render
                    if hasattr(self.eval_env, 'render_mode') and self.eval_env.render_mode == 'rgb_array_list':
                        frames = self.eval_env.render()
                        if frames and len(frames) > 0:
                            import imageio
                            video_path = os.path.join(self.video_dir, f'eval_{self.step}.mp4')
                            imageio.mimsave(video_path, frames, fps=30)
                            print(f"Video saved to {video_path}")
                except Exception as e:
                    print(f"Warning: Could not record video: {e}")

            average_episode_reward += episode_reward
        
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        return average_episode_reward


    def run(self):
        episode, episode_reward, terminated, truncated = 0, 0, True, False
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if terminated or truncated:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                # evaluate agent periodically
                if self.step > 0 and episode % self.cfg.save_summary_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    average_episode_reward = self.evaluate()
                    # lr_dict = {
                    #     'actor lr': self.agent.actor_optimizer.param_groups[0]['lr'],
                    #     'critic lr': self.agent.critic_optimizer.param_groups[0]['lr'],
                    #     'alpha lr': self.agent.log_alpha_optimizer.param_groups[0]['lr']
                    # }
                    average_episode_reward = save_summary(self.tb_writer, self.step, f'{self.cfg.env} train', average_episode_reward)
                
                # if self.step > 0 and episode % self.cfg.save_summary_freq:
                #     average_episode_reward = 0
                #     for episode in range(self.cfg.num_eval_episodes):
                #         obs, info = self.eval_env.reset()
                #         self.agent.reset()
                #         terminated, truncated = False, False
                #         episode_reward = 0
                #         while not (terminated or truncated): 
                #             with utils.eval_mode(self.agent):
                #                 action = self.agent.act(obs, sample=False)
                #             obs, reward, terminated, truncated, info = self.eval_env.step(action)
                #             episode_reward += reward

                #         average_episode_reward += episode_reward
                    
                #     average_episode_reward /= self.cfg.num_eval_episodes
                #     lr_dict = {
                #         'actor lr': self.agent.actor_optimizer.param_groups[0]['lr'],
                #         'critic lr': self.agent.critic_optimizer.param_groups[0]['lr'],
                #         'alpha lr': self.agent.log_alpha_optimizer.param_groups[0]['lr']
                #     }
                #     save_summary(self.tb_writer, lr_dict, self.step, f'{self.cfg.env} train', average_episode_reward)

                # save checkpoint periodically
                if self.step > 0 and episode % self.cfg.ckpt_frequency == 0:
                    self.save_checkpoint()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs, info = self.env.reset()
                self.agent.reset()
                terminated, truncated = False, False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)                    

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(terminated or truncated)
            # Get max episode steps (might need to adjust based on your environment)
            max_steps = info.get('TimeLimit.truncated', False) or truncated
            done_no_max = 0 if max_steps else float(terminated)

            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
        # when all done, save checkpoint
        self.save_checkpoint()


# @hydra.main(config_path='config/train.yaml', strict=True)
# @hydra.main(config_path='config/train_humanoid.yaml', strict=True)
# @hydra.main(config_path='config/train_hopper.yaml', strict=True)
@hydra.main(config_path='config/train_ant.yaml', strict=True)
def main(cfg):
    if not hasattr(cfg, 'ckpt_frequency'):
        cfg.ckpt_frequency = 100000  # 默认每1万步保存一次
    if not hasattr(cfg, 'resume_training'):
        cfg.resume_training = False  # 默认不从checkpoint恢复
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
