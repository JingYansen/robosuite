from collections import OrderedDict
import random
import numpy as np
import gym
import os

# import shutil
# from tensorboardX import SummaryWriter

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.environments.sawyer import SawyerEnv
from gym.envs.mujoco import mujoco_env
from gym import spaces

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from robosuite.models.arenas import BinSqueezeArena
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
    BananaObject,
    BowlObject,
)
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
    BananaVisualObject,
    BowlVisualObject,
)

from robosuite.models.tasks import BinSqueezeTask, UniformRandomSampler


class BinSqueeze(SawyerEnv, mujoco_env.MujocoEnv):
    def __init__(
            self,
            gripper_type="TwoFingerGripper",
            table_full_size=(0.39, 0.49, 0.82),
            table_target_size=(0.085, 0.085, 0.12),
            table_friction=(1, 0.005, 0.0001),
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=False,
            single_object_mode=0,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=1,
            horizon=1000,
            ignore_done=False,
            camera_name="targetview",
            camera_height=128,
            camera_width=128,
            camera_depth=False,

            video_height=256,
            video_width=256,
            render_drop_freq=0,
            hard_case = {
                'obj_names': ['Can', 'Can', 'Milk', 'Milk', 'Cereal'],
                'obj_poses': [
                    np.array([0.57, 0.405]),
                    np.array([0.63, 0.405]),
                    np.array([0.558, 0.35]),
                    np.array([0.642, 0.35]),
                    np.array([0.60, 0.36, 0.95]),
                ],
                'target_object': 'Cereal1'
            },
            total_steps=200,
            keys='image',
            action_pos_index=[0, 1, 2, 3, 4, 5, 6],
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with all types of objects.

                1: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            object_type (string): if provided, should be one of "milk", "bread", "cereal",
                or "can". Determines which type of object will be spawned on every
                environment reset. Only used if @single_object_mode is 2.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            action_pos_index:
                (x, y, z, u, v, w ,t)
                (0, 1, 2, 3, 4, 5, 6)
                eg: [2, 5] -> action space: (z, w)
        """

        # task settings
        self.obj_names = hard_case['obj_names']
        self.obj_poses = hard_case['obj_poses']
        self.target_object = hard_case['target_object']
        self.action_pos_index = action_pos_index

        assert len(self.obj_names) == len(self.obj_poses)
        assert len(self.action_pos_index) <= 7

        self.total_steps = total_steps
        self.cur_step = 0

        self.render_drop_freq = render_drop_freq
        self.video_height = video_height
        self.video_width = video_width

        self.single_object_mode = single_object_mode
        self.object_to_id = {"Milk": 0, "Bread": 1, "Cereal": 2, "Can": 3, "Banana": 4, "Bowl": 5}

        # settings for table top
        self.table_full_size = table_full_size
        self.table_target_size = table_target_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        if keys is None:
            assert self.use_object_obs, "Object observations need to be enabled."
            keys = ["image"]
        self.keys = keys

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        # information of objects
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

        # init objects position
        self._init_pos()

        # set up observation and action spaces
        flat_ob = self._flatten_obs(super().reset(), verbose=True)
        self.obs_dim = flat_ob.shape
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        high = np.ones(len(self.action_pos_index))
        low = -high.copy()
        self.action_space = spaces.Box(low=low, high=high)

    def reset(self):
        ob_dict = super().reset()
        return self._flatten_obs(ob_dict)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])

        return np.concatenate(ob_lst)

    def _name2obj(self, name):
        assert name in self.object_to_id.keys()

        if name is 'Milk':
            return MilkObject, MilkVisualObject
        elif name is 'Bread':
            return BreadObject, BreadVisualObject
        elif name is 'Cereal':
            return CerealObject, CerealVisualObject
        elif name is 'Can':
            return CanObject, CanVisualObject
        elif name is 'Banana':
            return BananaObject, BananaVisualObject
        elif name is 'Bowl':
            return BowlObject, BowlVisualObject

    def _make_objects(self):
        self.item_names = []
        self.ob_inits = []
        self.vis_inits = []

        idx = [0] * len(self.object_to_id)

        for name in self.obj_names:
            obj, vis_obj = self._name2obj(name)

            ix = self.object_to_id[name]
            idx[ix] += 1

            self.item_names.append(name + str(idx[ix]))
            self.ob_inits.append(obj)
            self.vis_inits.append(vis_obj)

        self.item_names_org = list(self.item_names)

    def _init_pos(self):
        for name, pos in zip(self.object_names, self.obj_poses):
            if len(pos) == 3:
                z = pos[2]
            else:
                z = 0.9
            self.teleport_object(name, pos[0], pos[1], z)

            if name == self.target_object:
                self.target_init_pos = pos

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = BinSqueezeArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.5, -0.3, 0])

        # make objects
        self._make_objects()

        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]), self.vis_inits[j]()))
        self.visual_objects = lst

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]), ob))

        self.mujoco_objects = OrderedDict(lst)
        self.object_names = list(self.mujoco_objects.keys())

        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = BinSqueezeTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
        )

        self.bin_pos = string_to_array(self.model.bin2_body.get("pos"))
        self.bin_size = self.table_target_size

    def clear_objects(self, obj):
        """
        Clears objects with name @obj out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.
        """
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name)[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def remove_object(self, obj):
        self.teleport_object(obj, x=10, y=10)

    def teleport_object(self, obj, x, y, z=0.9):
        """
        Teleport an object to a certain position (x, y, z).
        """
        assert obj in self.mujoco_objects.keys()

        sim_state = self.sim.get_state()
        x_dim = self.sim.model.get_joint_qpos_addr(obj)[0]
        y_dim = x_dim + 1
        z_dim = x_dim + 2

        sim_state.qpos[x_dim] = x
        sim_state.qpos[y_dim] = y
        sim_state.qpos[z_dim] = z

        self.sim.set_state(sim_state)
        self.sim.forward()

    def set_qpos(self, obj, qpos):
        """
        set qpos of object
        :param obj:
        :param qpos: (x, y, z, u, v, w, t)
        :return:
        """
        assert obj in self.mujoco_objects.keys()
        sim_state = self.sim.get_state()

        # set pos
        beg_dim = self.sim.model.get_joint_qpos_addr(obj)[0]
        dims = self.action_pos_index.copy() + beg_dim
        sim_state.qpos[dims] = qpos.copy()

        self.sim.set_state(sim_state)
        self.sim.forward()

    def _pre_action(self, action):
        beg_dim = self.sim.model.get_joint_qpos_addr('Milk1')[0]
        self.sim.data.qfrc_applied[beg_dim+2:beg_dim+7] += self.sim.data.qfrc_bias[beg_dim+2:beg_dim+7]

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        # teleport target object by (x, y, z, u, v, w, t)
        self.set_qpos(self.target_object, action)

        self.cur_step += 1
        self._pre_action(action)
        end_time = self.cur_time + self.control_timestep

        info = {}

        if self.render_drop_freq:
            i = 0
            info['image'] = []

        while self.cur_time < end_time:

            if self.render_drop_freq:
                if i % self.render_drop_freq == 0:
                    bird_image = self.sim.render(width=self.video_width, height=self.video_height, camera_name='birdview')
                    target_image = self.sim.render(width=self.video_width, height=self.video_height, camera_name='targetview')

                    info['image'].append(np.concatenate((bird_image, target_image), 1))

                i += 1

            self.sim.step()
            self.cur_time += self.model_timestep

        reward, done, _ = self._post_action(action)

        # done
        done = (self.cur_step >= self.total_steps)

        if done:
            print('Done!')

        ob_dict = self._get_observation()

        return self._flatten_obs(ob_dict), reward, done, info

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i])
            self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            # self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)

        # for checking distance to / contact with objects we want to pick up
        self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        # self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.ob_inits))
        self.objects_not_take = np.ones(len(self.ob_inits))

    def _reset_internal(self):
        super()._reset_internal()

        # reset positions of objects, and move objects out of the scene depending on the mode
        self._init_pos()

    def reward(self, action=None):
        # compute sparse rewards
        last_num = np.sum(self.objects_in_bins)
        self._check_success()
        reward = np.sum(self.objects_in_bins) - last_num


        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)

        # if reward != 0:
        print('Reward: ', reward)
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """
        pass

    def not_in_bin(self, obj_pos):

        bin_x_low = self.bin_pos[0] - self.bin_size[0] / 2
        bin_y_low = self.bin_pos[1] - self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0]
        bin_y_high = bin_y_low + self.bin_size[1]

        res = True
        if (
                obj_pos[2] > self.bin_pos[2]
                and obj_pos[0] < bin_x_high
                and obj_pos[0] > bin_x_low
                and obj_pos[1] < bin_y_high
                and obj_pos[1] > bin_y_low
                and obj_pos[2] < self.bin_pos[2] + self.bin_size[2]
        ):
            res = False
        return res

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        if self.use_camera_obs:

            bird_image = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='birdview')
            # target_image = np.rot90(self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='targetview'), 2)
            target_image = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='targetview')

            di["image"] = np.concatenate((bird_image, target_image), 1)

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                    or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i])
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int(
                (not self.not_in_bin(obj_pos)) and r_reach < 0.6
            )

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.ob_inits)

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba