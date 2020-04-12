import numpy as np
from collections import deque
from torchvision import transforms
from game2 import Game


class MaxAndSkipEnv:
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.canvas.c.shape, dtype=np.uint8)
        self._skip = skip
        self.env = env
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Grayscale(),
                                             transforms.Resize(size=(84, 84)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5], std=[0.20])
                                             ]
                                            )

    def warp_frame(self, frame):
        return self.transform(frame)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return self.warp_frame(max_frame), total_reward, done, info

    def reset(self, **kwargs):
        return self.warp_frame(self.env.reset(**kwargs))

    def stop(self, **kwargs):
        self.env.stop(**kwargs)


class FrameStack:
    def __init__(self, env, k=4):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

    def stop(self):
        self.env.stop()


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_env(draw=False):
    env = Game(draw)
    env = MaxAndSkipEnv(env)
    env = FrameStack(env)
    return env
