import gym
import numpy as np

import torch as th
from torch import nn
import pickle

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robogym.envs.rearrange import task_reach
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

class RobogymTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        randomized_task=None, #List of task identifiers/descriptions used
        mode="normal", #"normal" if normal mode, "green" if atsks using green color should be used
        all_descriptions = True #Whether to use all descriptions or only one description
    ):
        super().__init__(env)
        
        self.randomized_task = randomized_task
        
        # Load the task instruction sentence embeddings , either the normal ones or the ones describing green tasks
        if mode == "normal":
            with open("/home/frank/thesis/task_instructions/reach_task_embeddings.pickle", "rb") as f:
                self.task_embeddings = pickle.load(f)
        else:
            with open("/home/frank/thesis/task_instructions/reach_task_embeddings_green.pickle", "rb") as f:
                self.task_embeddings = pickle.load(f)
        self.mode = mode
        self.all_descriptions = all_descriptions
        self.observation_space = gym.spaces.Dict(
            spaces={
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(3, 50, 50)),
                "task": gym.spaces.Box(low=0, high=1, shape=(1536,)) # Task embeddings are 1536 long vectors
            }
        )

        self.reset() # Sample first goal


    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        # reward has shape [env_reward, goal_reward, success_reward]
        # env_reward contains potential penalties from the environment,
        # goal_reward contains intermediate rewards based on goal distance
        # success_reward contains rewards gained for completing a task
        reward = np.sum(rewards)
        observation = obs["vision_obs"][0]
        observation = np.moveaxis(observation, -1, 0) # Transform from channel last to channel first

        dict_obs = {"state": observation, "task":self.current_task}
        return dict_obs, reward, done, info

    def id_to_features(self, task_id):
        # Retrieve the action, color and shape corresponding to the task id
        # These can in turn be used to initialize the robogym environment
        # NOTE: pickup task is not used anymore, so all task_ids will be in [2,4,6,8]
        action = "pickup" if task_id in [1,3,5,7] else "reach"
        color = "red" if task_id in [1,2,3,4] else "blue"
        shape = "sphere" if task_id in [1,2,5,6] else "cube"
        return action, color, shape

    def reset(self, **kwargs):
        # Sample a new goal
        if self.randomized_task is None:
            # Sample a goal at random
            # Since pickup task was removed, tasks 1, 3, 5 and 7 are no longer used.
            # Only tasks 2, 4, 6 and 8 are used as unique tasks.
            # Different descriptions of the task can be chosen later in this function
            current_task = np.random.choice([2,4,6,8])
        else:
            #Only used during initial testing
            if len(self.randomized_task) == 1:
                current_task = self.randomized_task[0]
            #Only used during initial testing
            else:
                current_task = self.randomized_task[ np.random.randint(0, len(self.randomized_task)) ]
        #Identify features corresponding to the chosen task type
        action, color, shape = self.id_to_features(current_task)
        if self.mode == "green":
            #If green mode, overwrite the color of the target object
            color = "green"
        # Reset the robogym environment, setting the target object features
        self.env.reset(color1 = color, shape1 = shape, **kwargs)
        
        # Retrieve the vision observation
        observation = self.env.get_observation()[0]["vision_obs"][0]
        observation = np.moveaxis(observation, -1, 0)
        
        # Retrieve the task description observation (task_id, used for one-hot encoding)
        current_task_idx = int((current_task / 2) - 1)
        #current_task_idx now contains 0, 1, 2 or 3, corresponding to one of the four possible tasks
        current_task_idx = current_task_idx * 64 #Pick the first embedding matching the task type
        if self.mode == "green":
            #If green mode, set current_task_idx to the first embedding index corresponding to the chosen shape
            current_task_idx = 0 if shape == "sphere" else 64
        if self.all_descriptions == True:
            #Each task has 64 total descriptions, so pick one of them
            current_task_idx += np.random.randint(64) #Picks a number 0-63 (64 is excluded) to add to the idx
        self.current_task = self.task_embeddings[current_task_idx,:]
        dict_obs = {"state": observation, "task":self.current_task}
        return dict_obs
    

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_cnn_output=128, task_embedding_mlp=[64, 32], activation=th.nn.Tanh):
        self.state_cnn_output = state_cnn_output
        self.task_embedding_mlp = task_embedding_mlp
        self.activation = activation
        self._features_dim = state_cnn_output+task_embedding_mlp[-1]

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=self._features_dim)

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            if key == "state":
                n_input_channels = subspace.shape[0] #channels is the first value, thanks to transforming earlier
                
                # Vision network is adapted from the following article:
                #  Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
                #  Nature 518.7540 (2015): 529-533.
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                    )
                
                #Compute the shape
                with th.no_grad():
                    n_flatten = cnn(th.as_tensor(subspace.sample()[None]).float()).shape[1] #permute to fix channel position
                # End with one fully connected layer
                fc = nn.Sequential(nn.Linear(n_flatten, state_cnn_output), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key == "task":
                #mlp = [observation_space['task'].shape[0]] + self.task_embedding_mlp
                mlp = [subspace.shape[0]] + self.task_embedding_mlp
                layers = []
                for i in range(len(mlp)-1):
                    layers.append( nn.Linear(mlp[i], mlp[i+1]) )
                    layers.append( self.activation() )

                extractors[key] = nn.Sequential(*layers)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
    
def make_env(mode = "normal"):
    # Use mode = "green" for the evaluation environment to test performance on
    #  tasks where the target object is green (unseen in training)
    def _init():
        # Initialize the robogym environment.
        # 'vision': True - makes the robogym environment return the front view camera images as observation
        # 'max_timesteps_per_goal': 400 - Cuts off the simulation if it surpasses 400 timesteps, 
        #      resulting in lower reward in those cases, as only the goal_reward and env_reward penalty are gained.
        # 'vision_args': {'image_size': 50} - downsizes the observation images from [200,200] to [50,50] to greatly
        #       reduce training time.
        env = task_reach.make_env(constants={'vision': True, 'max_timesteps_per_goal': 400, 
                                  'vision_args': {'image_size': 50}})
        env = RobogymTaskWrapper(env, mode=mode)
        env.reset()
        return env
    return _init

if __name__ == '__main__':
    # Change n_envs value depending on available RAM, higher value speeds up training time but increases RAM usage
    n_envs = 8
    print("Starting procedure")
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward = True, clip_obs = 10., norm_obs_keys = ["state"])
    print("Environments created")
    # Runs the vision model (CNN) and task model (MLP) in parallel, and concatenates their final layers.
    # The DRL part of the network takes the concatenated layer as input, 
    #  attaches two output heads consiting of 2 layers of 64 units each, 
    #  and produces the appropriate output for the value function and the action space.
    policy_kwargs = dict(activation_fn = th.nn.Tanh,
                         features_extractor_class = CustomCombinedExtractor,
                         features_extractor_kwargs = dict(state_cnn_output = 128, task_embedding_mlp = [16,32], activation = th.nn.Tanh),
                         net_arch = dict(vf=[64,64], pi=[64, 64]))
    # Default values are used for the PPO hyperparameters
    model = PPO("MultiInputPolicy", env, policy_kwargs = policy_kwargs, verbose = 1, 
                n_steps = 1024, batch_size = 256, gamma = 0.999, tensorboard_log="tensorboard/")
    # Printing the model policy shows the layout of the entire model, including the vision and task models. 
    print(model.policy)
    print("Starting the learning of the PPO model")
    #Create green environment for the eval_callback function
    eval_env = SubprocVecEnv([make_env(mode = "green") for _ in range(n_envs//2)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env , norm_obs=True, norm_reward = True, clip_obs = 10., norm_obs_keys = ["state"])
    #Evaluate performance every 25k timesteps on the green environment
    eval_callback = EvalCallback(eval_env, log_path = "./thesis/logs/results/", eval_freq = 25000/(n_envs//2), n_eval_episodes=10)
    model.learn(250*4096, callback = eval_callback)
    print("Saving model and stats...")
    # Change paths as necessary
    model.save("./thesis/models/ppo_embeddings_all_desc")
    env.save("./thesis/models/vec_normalize_embeddings_all_desc")
