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
                 add_noise=False,
                 pixel_density=1):
        
        # Underlying MDP
        self.gridworld = GridWorld(rows, cols, randomize_starts, randomize_goals)
        
        # Observation params
        self.add_noise = add_noise
        self.pixel_density = pixel_density
        self.sensors = self._get_sensors()
        
        # Gym metadata
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        super().__init__()

    def _get_action_space(self):
        n_actions = len(self.gridworld.actions)
        return spaces.Discrete(n_actions)

    def _get_observation_space(self):
        assert isinstance(self.pixel_density, int), self.pixel_density
        def get_image_sensor():
            for sensor in self.sensors.sensors:
                if isinstance(sensor, ImageSensor):
                    return sensor
            raise NotImplementedError("GymEnv for pure GridWorld")
        return spaces.Box(low=0., high=1., shape=get_image_sensor().size, dtype=np.float32)

    def _get_sensors(self):
        img_sensor = ImageSensor(range=((0, self.gridworld._rows),
                                        (0, self.gridworld._cols)),
                                        pixel_density=self.pixel_density)
        if self.add_noise:
            return SensorChain([img_sensor, NoisySensor()])
        return SensorChain([img_sensor])
        
    def step(self, action):
        state, reward, done = self.gridworld.step(action)
        obs = self.sensors.observe(state)
        return obs, reward, done, dict(state=state)

    def reset(self):
        self.gridworld.reset()
        state = self.gridworld.get_state()
        return self.sensors.observe(state)
    
    def render(self, mode='human'):
        assert mode in ("human", "rgb_array"), mode

        if mode == "rgb_array":
            return self.sensors.observe(self.gridworld.get_state())
        
        raise NotImplementedError(mode)
        