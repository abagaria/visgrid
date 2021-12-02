import gym
import numpy as np
from gym import spaces
from .gridworld import GridWorld
from sensors import *


class GridWorldEnv(gym.Env):
    def __init__(self,
                 rows,
                 cols,
                 randomize_starts,
                 randomize_goals,
                 add_noise=False):
        assert rows == cols, "Should be square grid"
        
        # Underlying MDP
        self.gridworld = GridWorld(rows, cols, randomize_starts, randomize_goals)

        # Image dimensions
        img_width = img_height = 84
        self.scale = img_width // self.gridworld._rows
        
        # Observation params
        self.add_noise = add_noise
        self.sensors = self._get_sensors()
        
        # Gym metadata
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # Episode counter
        self.T = 0
        self.max_steps_per_episode = 100

        super().__init__()

    def _get_action_space(self):
        n_actions = len(self.gridworld.actions)
        return spaces.Discrete(n_actions)

    def _get_observation_space(self):
        def get_image_sensor():
            for sensor in self.sensors.sensors:
                if isinstance(sensor, ImageSensor):
                    return sensor
            raise NotImplementedError("GymEnv for pure GridWorld")
        return spaces.Box(low=0., high=1., shape=get_image_sensor().size, dtype=np.float32)

    def _get_sensors(self):
        if self.add_noise:
            sensor_list = [
                OffsetSensor(offset=(0.5, 0.5)),
                NoisySensor(sigma=0.05),
                ImageSensor(range=((0, self.gridworld._rows),
                                   (0, self.gridworld._cols))),
                ResampleSensor(scale=self.scale),
                BlurSensor(sigma=0.6, truncate=1.),
                NoisySensor(sigma=0.01)
            ]
        else:
            sensor_list = [
                OffsetSensor(offset=(0.5, 0.5)),
                ImageSensor(range=((0, self.gridworld._rows),
                                   (0, self.gridworld._cols))),
                ResampleSensor(scale=self.scale),
            ]
        return SensorChain(sensor_list)
        
    def step(self, action):
        self.T += 1
        state, reward, done = self.gridworld.step(action)
        obs = self.sensors.observe(state)
        reset = self.T % self.max_steps_per_episode == 0
        return obs, reward, done or reset, dict(
            state=state, 
            needs_reset=reset
        )

    def reset(self):
        self.T = 0
        self.gridworld.reset()
        state = self.gridworld.get_state()
        return self.sensors.observe(state)
    
    def render(self, mode='human'):
        assert mode in ("human", "rgb_array"), mode

        if mode == "rgb_array":
            return self.sensors.observe(self.gridworld.get_state())
        
        raise NotImplementedError(mode)
        