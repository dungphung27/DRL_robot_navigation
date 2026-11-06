import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import copy 
import matplotlib.pyplot as plt
import os
from report_plots import export_training_report
from replay_buffer import Batch, ReplayMemory
from exp_schedule import ExponentialSchedule
from envs.generic_env import GenericWorld
from envs.obstacles_env import ObstaclesWorld
from datetime import datetime
from pathlib import Path
def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # YOUR CODE HERE:  define the layers of your model such that
        # * there are `num_layers` nn.Linear modules / layers
        # * all activations except the last should be ReLU activations
        #   (this can be achieved either using a nn.ReLU() object or the nn.functional.relu() method)
        # * the last activation can either be missing, or you can use nn.Identity()
        self.layers = [(nn.Linear(self.state_dim, hidden_dim)).to(self.device), nn.ReLU().to(self.device)]

        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim).to(self.device))
            self.layers.append(nn.ReLU().to(self.device))

        # last one
        self.layers.append(nn.Linear(hidden_dim, self.action_dim).to(self.device))

        self.linear_stack = nn.Sequential(*self.layers)
        
    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        # YOUR CODE HERE:  use the defined layers and activations to compute
        # the action-values tensor associated with the input states.
        
        result = states
        for layer in self.layers:
            result = layer(result).to(self.device)
        return result
    
    
    # utility methods for cloning and storing models.  DO NOT EDIT
    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }


# test code, do not edit


# def _test_dqn_forward(dqn_model, input_shape, output_shape):
#     """Tests that the dqn returns the correctly shaped tensors."""
#     inputs = torch.torch.randn((input_shape))
#     outputs = dqn_model(inputs)

#     if not isinstance(outputs, torch.FloatTensor):
#         raise Exception(
#             f'DQN.forward returned type {type(outputs)} instead of torch.Tensor'
#         )

#     if outputs.shape != output_shape:
#         raise Exception(
#             f'DQN.forward returned tensor with shape {outputs.shape} instead of {output_shape}'
#         )

#     if not outputs.requires_grad:
#         raise Exception(
#             f'DQN.forward returned tensor which does not require a gradient (but it should)'
#         )


# dqn_model = DQN(10, 4)
# _test_dqn_forward(dqn_model, (64, 10), (64, 4))
# _test_dqn_forward(dqn_model, (2, 3, 10), (2, 3, 4))
# del dqn_model

# dqn_model = DQN(64, 16)
# _test_dqn_forward(dqn_model, (64, 64), (64, 16))
# _test_dqn_forward(dqn_model, (2, 3, 64), (2, 3, 16))
# del dqn_model

# testing custom dump / load
# dqn1 = DQN(10, 4, num_layers=10, hidden_dim=20)
# dqn2 = DQN.custom_load(dqn1.custom_dump())
# assert dqn2.state_dim == 10
# assert dqn2.action_dim == 4
# assert dqn2.num_layers == 10
# assert dqn2.hidden_dim == 20

def train_dqn_batch(optimizer: torch.optim.Optimizer, batch: Batch, dqn_model: nn.Module, dqn_target: nn.Module, gamma: float) -> float:
    """Perform a single batch-update step on the given DQN model.

    :param optimizer: nn.optim.Optimizer instance.
    :param batch:  Batch of experiences (class defined earlier).
    :param dqn_model:  The DQN model to be trained.
    :param dqn_target:  The target DQN model, ~NOT~ to be trained.
    :param gamma:  The discount factor.
    :rtype: float  The scalar loss associated with this batch.
    """
    # YOUR CODE HERE:  compute the values and target_values tensors using the
    # given models and the batch of data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    values : torch.Tensor = dqn_model(batch.states.to(device)).gather(1, batch.actions.to(device)).to(device)

    # Compute the target values tensor
    with torch.no_grad():
        next_state_values = torch.zeros(len(batch.next_states.to(device))).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s.item() is False, batch.dones.to(device))), dtype=torch.bool).to(device)
        next_state_values[non_final_mask] = dqn_target(batch.next_states.to(device)[non_final_mask]).to(device).max(1)[0].detach().to(device)

        target_values = batch.rewards.to(device) + gamma * next_state_values.unsqueeze(1).to(device)
    
    # DO NOT EDIT FURTHER

    assert (
        values.shape == target_values.shape
    ), 'Shapes of values tensor and target_values tensor do not match.'

    # testing that the value tensor requires a gradient,
    # and the target_values tensor does not
    assert values.requires_grad, 'values tensor should not require gradients'
    assert (
        not target_values.requires_grad
    ), 'target_values tensor should require gradients'

    # computing the scalar MSE loss between computed values and the TD-target
    loss = F.mse_loss(values, target_values).to(device)

    optimizer.zero_grad()  # reset all previous gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # perform one gradient descent step

    return loss.item()


def train_dqn(
    env : GenericWorld,
    observation_space, 
    action_space,
    num_steps,
    *,  
    checkpoint_dir=None,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    replay_policy=None,
    batch_size,
    exploration : ExponentialSchedule,
    gamma : float,
    render: bool = True
):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_steps: Total number of steps to be used for training
        - num_saves: How many models to save to analyze the training progress.
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate
                                    the memory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule
        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints")
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # get the state_size from the environment
    state_size = observation_space

    # initialize the DQN and DQN-target models
    dqn_model = DQN(state_size, action_space).to(device)
    dqn_target = DQN.custom_load(dqn_model.custom_dump()).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps, replay_policy)

    # initiate lists to store returns, lengths and losses
    rewards = []
    returns = []
    lengths = []
    losses = []
    outcomes = []
    total_rewards = []
    G = 0 # discounted returns

    # initiate structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # use this to indicate the index of the current episode
    t_episode = 0  # use this to indicate the time-step inside current episode
    max_episode_steps = 1000
    state = env.reset(render=True)  # initialize state of first episode

    # iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps, ncols=100)
    for t_total in pbar:
        # use t_total to indicate the time-step from the beginning of training

        # save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        # YOUR CODE HERE:
        #  * sample an action from the DQN using epsilon-greedy
        #  * use the action to advance the environment by one step
        #  * store the transition into the replay memory
        
        eps = exploration.value(t_total)
        action = None
        if (np.random.random() < eps):
            action = np.random.randint(action_space)
        else:
            action = torch.argmax(dqn_model(torch.Tensor(state).to(device)), dim=0).to(device).item()
        
        next_state, reward, done = env.step(action, render=render)
        # Episode đã đến giới hạn bước chưa?
        timeout = (t_episode + 1 >= max_episode_steps)

        # Giữ lại nguyên nhân kết thúc từ môi trường
        terminated = done

        # Kết thúc episode khi đạt giới hạn
        done = terminated or timeout
        if done:
            if terminated:
                if env._within_goal():
                    end_reason = "SUCCESS"
                else:
                    end_reason = "COLLISION"
            else:
                end_reason = "TIMEOUT"

            print(
                f"Episode {i_episode} | "
                f"Steps: {t_episode + 1} | "
                f"Result: {end_reason}"
            )
            outcomes.append(end_reason)
        rewards.append(reward)
        
        G = reward + gamma * G
        memory.add(state, action, reward, next_state, done)
        state = next_state
        
        # YOUR CODE HERE:  once every 4 steps,
        #  * sample a batch from the replay memory
        #  * perform a batch update (use the train_dqn_batch() method!)
        
        if (t_total % 4 == 0):
            sampled_batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, sampled_batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # YOUR CODE HERE:  once every 10_000 steps,
        #  * update the target network (use the dqn_model.state_dict() and
        #    dqn_target.load_state_dict() methods!)
        
        if ((t_total + 1) % 10_000 == 0):
            dqn_target.load_state_dict(dqn_model.state_dict())
            # Lưu checkpoint mỗi 100.000 steps
        if (t_total + 1) % 100_000 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{t_total + 1}.pt"

            torch.save(
                {
                    "step": t_total + 1,
                    "model": dqn_model.custom_dump(),
                    "target_state_dict": dqn_target.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path
            )

            print(f"\nSaved checkpoint: {checkpoint_path}")

        if done:
            # Tính discounted return của episode vừa kết thúc
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G

            # Lưu kết quả
            lengths.append(t_episode + 1)
            returns.append(G)

            pbar.set_description(
                f'Episode: {i_episode} | '
                f'Steps: {t_episode + 1} | '
                f'Return: {G:5.2f} | '
                f'Epsilon: {eps:4.2f}'
            )

            # Reset môi trường để bắt đầu episode mới
            state = env.reset(render=render)

            # Reset các biến thống kê
            t_episode = 0
            i_episode += 1
            G = 0.0
            total_rewards.append(sum(rewards))
            rewards = []
        else:
            t_episode += 1

    saved_models['100_0'] = copy.deepcopy(dqn_model)

    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
        outcomes,
        np.array(total_rewards) 
    )

def plot(returns, lengths, losses):
    ### YOUR PLOTTING CODE HERE
    plt.plot(range(len(returns)), returns, color='red', alpha=0.5, label='raw return data')
    plt.plot(range(len(returns)), rolling_average(data=returns, window_size=10), color='red', label='smooth return data')
    plt.title("Autonomous Nav: Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Discounted Return")
    plt.legend()
    plt.show()
    
    if (len(losses) != 0):
        plt.plot(range(len(losses)), losses, color='blue', alpha=0.5, label='raw loss data')
        plt.plot(range(len(losses)), rolling_average(data=losses, window_size=10), color='blue', label='smooth loss data')
        plt.title("Autonomous Nav: Loss per 4 Steps")
        plt.xlabel("4 Steps")
        plt.ylabel("Loss value")
        plt.legend()
        plt.show()
    
    plt.plot(range(len(lengths)), lengths, color='green', alpha=0.5, label='raw length data')
    plt.plot(range(len(lengths)), rolling_average(data=lengths, window_size=10), color='green', label='smooth length data')
    plt.title("Autonomous Nav: Epsiode Length per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.legend()
    plt.show()

def save(filename, returns, lengths, losses):
    f = open(filename, 'w')

    f.write('Returns\n')
    f.write(str(returns))
    f.write('\n')

    f.write('Lengths\n')
    f.write(str(lengths))
    f.write('\n')

    f.write('Losses\n')
    f.write(str(losses))
    f.write('\n')

    f.close()

if __name__ == "__main__":
    env = ObstaclesWorld(300, 300, see_all=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    checkpoint_dir = Path("checkpoints") / f"checkpoint_{timestamp}"

    report_dir = Path("results") / f"report_{timestamp}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint folder: {checkpoint_dir}")
    print(f"Report folder: {report_dir}")
    gamma = 0.99

    # we train for many time-steps;  as usual, you can decrease this during development / debugging.
    # but make sure to restore it to 1_500_000 before submitting.
    num_steps = 1_000_000
    num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 30_000
    replay_prepopulate_steps = 0 #10 #50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.01, 400_000)

    # this should take about 90-120 minutes on a generic 4-core laptop
    dqn_models, returns, lengths, losses, outcomes, total_rewards = train_dqn(
        env,
        observation_space=64,
        checkpoint_dir=checkpoint_dir,
        action_space=3,
        num_steps=num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        replay_policy=None,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
        render=True
    )
    os.makedirs("checkpoints", exist_ok=True)

        # Lưu các model vào thư mục của lần training hiện tại
    for model_name, model in dqn_models.items():
        model_path = checkpoint_dir / f"dqn_{model_name}.pth"

        torch.save(
            model.state_dict(),
            model_path
        )

        print(f"Saved model: {model_path}")
    export_training_report(
        returns=returns,
        lengths=lengths,
        losses=losses,
        outcomes=outcomes,
        total_rewards=total_rewards,
        folder=str(report_dir),
        window=50
    )

    print("Đã lưu biểu đồ và dữ liệu báo cáo!")
    print("Đã lưu tất cả các mô hình DQN!")
    save(
        str(report_dir / "training_metrics.txt"),
        returns,
        lengths,
        losses
    )
    plot(returns, lengths, losses)
    # assert len(dqn_models) == num_saves
    # assert all(isinstance(value, DQN) for value in dqn_models.values())

    # saving computed models to disk, so that we can load and visualize them later.
    # checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
    # torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')