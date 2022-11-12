import argparse
import copy
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from deep_q_network import DeepQNetwork
from tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=2048, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=50000)
    parser.add_argument("--num_epochs", type=int, default=500000)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=204800,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    env.reset()
    model = model.to(device)

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    aver = 0
    while epoch < opt.num_epochs:
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon

        state = env.get_simple_image().to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(state[None, :])[0]
        model.train()
        if random_action:
            action = randint(0, 39)
        else:
            action = torch.argmax(predictions).item()

        reward, done, next_state = env.step((action // 4, action % 4), render=False)

        next_state = next_state.to(device)
        replay_memory.append([state, action, reward, next_state, done])

        if done:
            final_score = env.score
            if aver == 0:
                aver = final_score
            else:
                aver = aver * 0.999 + final_score * 0.001
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            env.reset()
        else:
            continue
        if len(replay_memory) < opt.batch_size * 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(np.array(action_batch, dtype=np.int64)[:, None])
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        q_values = model(state_batch)
        q_values = q_values.gather(1, action_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        next_prediction_batch = torch.max(next_prediction_batch, 1).values.reshape(-1, 1)
        done_batch = torch.tensor(done_batch).reshape(-1, 1)

        tmp = tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                    zip(reward_batch, done_batch, next_prediction_batch))

        y_batch = torch.cat(tmp)[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("\rEpoch: {:6}/{}, Score: {:4}, Cleared lines: {:4}, Average score: {:8}".format(
            epoch,
            opt.num_epochs,
            final_score,
            final_cleared_lines,
            format(aver, '.2f')), end="")
        if epoch % 1000 == 0:
            print()
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
