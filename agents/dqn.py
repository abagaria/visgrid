import pfrl
import numpy as np
import torch.nn as nn
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from gridworld.gridenv import GridWorldEnv


env = GridWorldEnv(6, 6, True, False, False, 14)
n_actions = env.action_space.n

def phi(x):
    return np.asarray(x, dtype=np.float32)[None, ...]

q_func = nn.Sequential(
            pnn.SmallAtariCNN(n_input_channels=1), # Batch, width, height (no frame stack)
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
)

explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            0.0,
            1000,
            lambda: np.random.randint(n_actions),
)

opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
        q_func.parameters(),
        lr=2.5e-4,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
)

rbuf = replay_buffers.PrioritizedReplayBuffer(
            10 ** 4,
            alpha=0.6,
            beta0=0.4,
            betasteps=10000,
            num_steps=3
)

agent = agents.DQN(q_func,
        opt,
        rbuf,
        gpu=-1,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=100,
        target_update_interval=500,
        update_interval=1,
        batch_accumulator="sum",
        phi=phi
    )

experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=10000,
            eval_n_steps=None,
            eval_n_episodes=1,
            eval_interval=1000,
            outdir="/tmp/",
            save_best_so_far_agent=False,
            eval_env=env,
)
