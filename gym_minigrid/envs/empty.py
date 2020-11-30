from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        grid_width=None,
        grid_height=None,
        agent_start_pos=(1, 1),
        random_start_pos=False,
        agent_start_dir=0,
        step_reward=0,
        final_reward=1,
        obstacle_type=None,
        gap_pos_list=None,
        extra_goal=None
    ):
        self.agent_start_pos = agent_start_pos
        self.random_start_pos = random_start_pos
        self.agent_start_dir = agent_start_dir
        self.obstacle_type = obstacle_type
        self.gap_pos_list = gap_pos_list
        self.extra_goal = extra_goal
        if grid_height == None and grid_width == None:
            grid_height = size
            grid_width = size

        super().__init__(
            height=grid_height,
            width=grid_width,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            step_reward=step_reward,
            final_reward=final_reward
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        if self.extra_goal:
            self.put_obj(Goal2(), self.extra_goal, height - 2)

        # Place the agent
        if self.random_start_pos:
            self.agent_pos = (random.randint(1, width - 2),
                              random.randint(1, height - 2))
            self.agent_dir = self.agent_start_dir
        elif self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


        if self.obstacle_type:
            # Place the obstacle
            assert self.gap_pos_list, 'You must specify a gap_pos where you ' \
                                   'would ' \
                                 'like the obstacle to be placed'
            for gap_pos in self.gap_pos_list:
                self.put_obj(self.obstacle_type(), gap_pos[0], gap_pos[1])
            self.mission = (
                "avoid the lava and get to the green goal square"
                if self.obstacle_type == Wall
                else "find the opening and get to the green goal square"
            )
        else:
            self.mission = "get to the green goal square"

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)


class EmptyEnv1x3(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=3, grid_width=5,
                         random_start_pos=False, agent_start_pos=(1, 1),
                         **kwargs)

class EmptyEnv1x3RSS(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=3, grid_width=5,
                         random_start_pos=True, agent_start_pos=(1, 1),
                         **kwargs)

class EmptyEnv1x10(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(grid_height=3, grid_width=12,
                         random_start_pos=True,
                         **kwargs)

class EmptyEnv5x5R01(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, step_reward=0, final_reward=1, **kwargs)

class EmptyEnv5x5R01Wall1(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(5, 3)], **kwargs)

class EmptyEnv5x5R01Wall2(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(5, 3), (4, 3)],
                         **kwargs)

class EmptyEnv5x5R01Wall3(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(5, 3), (4, 3), (3,
                                                                          3)],
                         **kwargs)

class EmptyEnv5x5Rn10(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, step_reward=-1, final_reward=0, **kwargs)

class EmptyEnv10x10R01(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, step_reward=0, final_reward=1, **kwargs)

class EmptyEnv10x10R01Wall1(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(10, 3)], **kwargs)

class EmptyEnv10x10R01Wall2(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(10, 3), (9, 3)],
                         **kwargs)

class EmptyEnv10x10R01Wall3(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(10, 3), (9, 3),
                                                           (8, 3)],
                         **kwargs)

class EmptyEnv10x10Rn10(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12, step_reward=-1, final_reward=0, **kwargs)

class EmptyEnv20x20R01(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=22, step_reward=0, final_reward=1, **kwargs)


class EmptyEnv20x20R01Wall1(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=22, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(20, 3)], **kwargs)


class EmptyEnv20x20R01Wall2(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=22, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(20, 3), (19, 3)],
                         **kwargs)


class EmptyEnv20x20R01Wall3(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=22, step_reward=0, final_reward=1,
                         obstacle_type=Wall, gap_pos_list=[(20, 3), (19, 3),
                                                           (18, 3)],
                         **kwargs)

class EmptyEnv20x20Rn10(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=22, step_reward=-1, final_reward=0, **kwargs)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)


register(
    id='MiniGrid-Empty-1x3-v0',
    entry_point='gym_minigrid.envs:EmptyEnv1x3'
)

register(
    id='MiniGrid-Empty-RandStartState-1x3-v0',
    entry_point='gym_minigrid.envs:EmptyEnv1x3RSS'
)

register(
    id='MiniGrid-Empty-1x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv1x10'
)

register(
    id='MiniGrid-Empty-Reward-0-1-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5R01'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall1-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5R01Wall1'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall2-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5R01Wall2'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall3-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5R01Wall3'
)

register(
    id='MiniGrid-Empty-Reward--1-0-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5Rn10'
)

register(
    id='MiniGrid-Empty-Reward-0-1-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv10x10R01'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall1-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv10x10R01Wall1'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall2-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv10x10R01Wall2'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall3-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv10x10R01Wall3'
)

register(
    id='MiniGrid-Empty-Reward--1-0-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv10x10Rn10'
)

register(
    id='MiniGrid-Empty-Reward-0-1-20x20-v0',
    entry_point='gym_minigrid.envs:EmptyEnv20x20R01'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall1-20x20-v0',
    entry_point='gym_minigrid.envs:EmptyEnv20x20R01Wall1'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall2-20x20-v0',
    entry_point='gym_minigrid.envs:EmptyEnv20x20R01Wall2'
)

register(
    id='MiniGrid-Empty-Reward-0-1-Wall3-20x20-v0',
    entry_point='gym_minigrid.envs:EmptyEnv20x20R01Wall3'
)

register(
    id='MiniGrid-Empty-Reward--1-0-20x20-v0',
    entry_point='gym_minigrid.envs:EmptyEnv20x20Rn10'
)
#