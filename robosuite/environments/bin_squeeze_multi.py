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
from random import choice

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


class BinSqueezeMulti(SawyerEnv, mujoco_env.MujocoEnv):
    def __init__(
            self,
            gripper_type="TwoFingerGripper",
            table_full_size=(0.39, 0.49, 0.82),
            table_target_size=(0.105, 0.085, 0.12),
            table_friction=(1, 0.005, 0.0001),
            use_camera_obs=True,
            use_object_obs=True,
            single_object_mode=0,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            camera_name="targetview",
            camera_height=64,
            camera_width=64,
            camera_depth=False,
            render_drop_freq=0,
            obj_names=['Can'] * 3 + ['Milk'] * 3 + ['Bread'] * 3 + ['Cereal'] * 3,
            place_num=5,
            obj_poses=np.array([
                np.array([-0.03, 0.03, 0]),
                np.array([0.03, 0.03, 0]),
                np.array([-0.03, -0.03, 0]),
                np.array([0.03, -0.03, 0])
            ]),
            target_init_pos=np.array([0, 0, 0.17]),
            total_steps=1000,
            step_size=0.002,
            angle_scale=0.08,
            orientation_scale=0.08,
            energy_tradeoff=0,
            neg_ratio=10,
            force_ratios=[3, 3, 0.3],
            z_limit=0.21,
            keys='image',
            fix_rotation=False,
            no_delta=False,
            random_quat=False,
            random_target=True,
            test_cases=[],
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
        self.obj_names = obj_names
        self.obj_poses = obj_poses
        self.target_init_pos = target_init_pos
        self.place_num = place_num
        self.initialize_objects = False
        self.fix_rotation = fix_rotation
        self.no_delta = no_delta
        self.random_quat = random_quat
        self.random_target = random_target

        if self.fix_rotation:
            self.action_dim = 3
        else:
            self.action_dim = 7
        self.stack = []

        assert self.place_num <= len(self.obj_names)

        self.total_steps = total_steps
        self.step_size = step_size
        self.angle_scale = angle_scale
        self.orientation_scale = orientation_scale
        self.energy_tradeoff = energy_tradeoff
        self.force_ratios = force_ratios
        self.z_limit = z_limit
        self.cur_step = 0
        self.total_reward = 0
        self.over_times = 0
        self.success_objs = 0

        self.render_drop_freq = render_drop_freq

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

        # reward configuration
        self.neg_ratio = neg_ratio

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

        # set up observation and action spaces
        flat_ob = self._flatten_obs(super().reset(), verbose=True)
        self.obs_dim = flat_ob.shape
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        high = np.ones(self.action_dim)
        low = -high.copy()
        self.action_space = spaces.Box(low=low, high=high)

    def reset(self):
        ob_dict = super().reset()
        return self._flatten_obs(ob_dict)

    def _choose_target(self):
        self.target_object = choice(self.object_to_choose)

    def _finish_a_target(self):
        self.object_to_choose.remove(self.target_object)

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
        self.object_to_choose = self.object_names.copy()

        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = BinSqueezeTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
            self.obj_poses
        )
        # self.model.place_objects()

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

    def teleport_object(self, obj, x, y, z=0.95, uvwt=None):
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

        if uvwt is not None:
            assert len(uvwt) == 4
            beg_dim = z_dim + 1
            for i, val in enumerate(uvwt):
                sim_state.qpos[beg_dim+i] = val

        self.sim.set_state(sim_state)
        self.sim.forward()

    def get_abs_pos(self, obj, relative_pos):
        bottom_offset = self.mujoco_objects[obj].get_bottom_offset()
        pos = self.model.bin2_offset - bottom_offset + relative_pos

        return pos

    def prepare_objects(self):
        self._choose_target()

        self._pre_action(None)
        for _ in range(100):
            self.sim.step()
        self._post_action(None)

        target_obj = self.target_object
        pos = self.get_abs_pos(target_obj, self.target_init_pos)
        self.teleport_object(target_obj, pos[0], pos[1], pos[2], uvwt=[1., 0., 0., 0.])

        self.initialize_objects = True

    def get_tar_obj_pos(self):
        beg_dim, end_dim = self.sim.model.get_joint_qpos_addr(self.target_object)
        return self.sim.get_state().qpos[beg_dim : end_dim].copy()

    def step_obj_by_action(self, obj, action):
        """
        change qpos of obj by delta
        :param obj:
        :param action: (x, y, z, u, v, w, t)
        :return:
        """
        assert obj in self.mujoco_objects.keys()
        sim_state = self.sim.get_state()

        # set pos
        beg_dim, end_dim = self.sim.model.get_joint_qpos_addr(obj)

        # change cur pos

        self.target_cur_pos[0:3] += action[0:3].copy()

        theta = np.arccos(self.target_cur_pos[3].copy()) * 2
        S = np.sin(theta / 2)
        if S == 0:
            X, Y, Z = 0., 0., 0.
        else:
            X, Y, Z = self.target_cur_pos[4:7].copy() / S

        delta_X, delta_Y, delta_Z = action[4:7].copy()
        angle = self._angle(np.array([X, Y, Z]), np.array([delta_X, delta_Y, delta_Z]))

        # (u, v, w, t) is world rotation
        if self.no_delta:
            new_X, new_Y, new_Z = self._normalize(np.array([delta_X, delta_Y, delta_Z]))
        # (u, v, w, t) is the delta rotation
        else:
            new_X, new_Y, new_Z = self._normalize(np.array([X + delta_X, Y + delta_Y, Z + delta_Z]))

        new_theta = theta + action[3].copy()
        if new_theta >= 2 * np.pi and action[3] > 0:
            new_theta -= 2 * np.pi
        elif new_theta <= 0 and action[3] < 0:
            new_theta += 2 * np.pi

        new_C = np.cos(new_theta / 2)
        new_S = np.sin(new_theta / 2)

        self.target_cur_pos[3:7] = np.array([new_C, new_X * new_S, new_Y * new_S, new_Z * new_S])

        # set pos index in sim
        sim_state.qpos[beg_dim : end_dim] = self.target_cur_pos.copy()

        # set vel
        beg_dim, end_dim = self.sim.model.get_joint_qvel_addr(obj)
        sim_state.qvel[beg_dim : end_dim] = 0

        self.sim.set_state(sim_state)
        self.sim.forward()

        info = {'angle': angle}
        return info

    def _normalize(self, x):
        if len(x) > 1:
            x_norm = np.linalg.norm(x)
            if x_norm > 0:
                x = x / x_norm
        else:
            x = np.sign(x)

        return x

    def _sigmoid(self, x, b=0.):
        x = 1. / (1 + np.exp(-x)) - b
        return x

    def _angle(self, vec1, vec2):
        v1 = vec1.copy()
        v2 = vec2.copy()

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 == 0 or len_v2 == 0:
            return 0

        cos_angle = v1.dot(v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        angle = angle * 360 / 2 / np.pi

        return angle

    def _norm_action(self, old_action):
        action = old_action.copy()

        ## normalize coord
        coordinates = old_action[0:3].copy()
        action[0:3] = self._normalize(coordinates) * self.step_size

        ## normalize orientation
        theta = old_action[3].copy()
        action[3] = self._sigmoid(theta, 0.5) * 2 * np.pi * self.angle_scale

        xyz = old_action[4:7].copy()
        action[4:7] = self._normalize(xyz) * self.orientation_scale

        return action

    def _pre_action(self, action):
        if action is None:
            # gravity compensation
            self.sim.data.qfrc_applied[
                self._ref_joint_vel_indexes
            ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]
            return

        gripper_dim = self.sim.model.get_joint_qpos_addr(self.object_names[0])[0]
        beg_dim = gripper_dim + self.object_names.index(self.target_object) * 6
        ## set force
        sigs = np.sign(action[0:3])

        self.sim.data.qfrc_applied[beg_dim:beg_dim+3] = self.sim.data.qfrc_bias[beg_dim:beg_dim+3] * ( 1 + self.force_ratios * sigs)
        # self.sim.data.qfrc_applied[beg_dim+2:beg_dim+6] = self.sim.data.qfrc_bias[beg_dim+2:beg_dim+6]

    def _post_action(self, action, info={}):
        if action is None:
            sim_state = self.sim.get_state()
            sim_state.qvel[:] = 0
            self.sim.set_state(sim_state)
            self.sim.forward()
            return

        # remove vel
        sim_state = self.sim.get_state()
        beg_dim, end_dim = self.sim.model.get_joint_qvel_addr(self.target_object)
        sim_state.qvel[beg_dim : end_dim] = 0

        self.sim.set_state(sim_state)
        self.sim.forward()

        # calculate reward
        reward, done = self.reward(action, info)

        # done
        self.cur_step += 1
        if done:
            print('This Done!')

        return reward, done, info

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        info = {}

        ## fix rotation
        if self.fix_rotation:
            temp = action.copy()
            action = np.zeros(7)
            action[0:3] = temp

        ## prepare
        if not self.initialize_objects:
            self.prepare_objects()

        ## pre action: remove gravity and other forces.
        info.update({'theta': action[3].copy(), 'old_action': action.copy()})
        self._pre_action(action)

        ## get cur pos
        self.target_cur_pos = self.get_tar_obj_pos()

        ## teleport target object by (x, y, z, u, v, w, t)
        action = self._norm_action(action)
        info.update({'norm_action': action.copy()})

        temp_info = self.step_obj_by_action(self.target_object, action)
        info.update(temp_info)

        ## mujoco step
        end_time = self.cur_time + self.control_timestep
        while self.cur_time < end_time:
            self.sim.step()
            self.cur_time += self.model_timestep

        ## post action: calculate reward
        reward, this_done, info = self._post_action(action, info)
        self.total_reward += reward
        if this_done:
            self.success_objs += 1
            ## finish
            if self.success_objs == self.place_num:
                done = True
            ## next
            else:
                ## if out of bound clear this
                if reward < -10:
                    self.remove_object(self.target_object)

                ## set next
                self.initialize_objects = False
                done = False
        else:
            if self.cur_step >= self.total_steps:
                done = True
            else:
                done = False

        if done:
            print('All done!')
            info['stack_len'] = len(self.stack)
            info['total_reward'] = self.total_reward
            info['num_steps'] = self.cur_step
            if self.success_objs == self.place_num:
                info['num_steps_succ'] = self.cur_step
                info['succ'] = 1
            else:
                info['num_steps_fail'] = self.cur_step
                info['succ'] = 0
        ## obs
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

        # reset cur step
        self.cur_step = 0
        self.total_reward = 0
        self.success_objs = 0

        self.initialize_objects = False
        for obj in self.object_names:
            self.remove_object(obj)

    def reward(self, action=None, info={}):
        # get z pos
        target_pos = self.get_tar_obj_pos()
        z_pos = target_pos[2]

        # energy
        energy = self.energy_tradeoff * (np.square(info['theta']))
        assert energy <= self.energy_tradeoff * 2 and energy >= 0

        done = False

        # not in bin
        if self.not_in_bin(target_pos[0:3]):
            reward = -10 - self.neg_ratio * (self.total_steps - self.cur_step)
            done = True
        else:
            # get obj mjcf
            target_obj_mjcf = self.mujoco_objects[self.target_object]
            bottom_offset = target_obj_mjcf.get_bottom_offset()

            # calculate z offset relative to bin
            z_pos_to_bin = z_pos - (self.model.bin2_offset[2] - bottom_offset[2])
            epsilon = 2e-4

            # out of z
            if z_pos_to_bin >= self.z_limit:
                reward = -10 - self.neg_ratio * (self.total_steps - self.cur_step)
                done = True
            # success
            elif z_pos_to_bin <= epsilon:
                reward = 100
                done = True
            # above
            else:
                delta = (self.z_limit - z_pos_to_bin) / self.z_limit
                reward = delta ** 2 - 1
                done = False

        reward -= energy
        print('Reward: ', reward)
        # float overflow
        assert reward <= 100
        return reward, done

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
        delta_x = 0.03
        delta_y = 0.02
        delta_z = 0.01

        if (
                obj_pos[2] > self.bin_pos[2] - delta_z
                and obj_pos[0] < bin_x_high + delta_x
                and obj_pos[0] > bin_x_low - delta_x
                and obj_pos[1] < bin_y_high + delta_y
                and obj_pos[1] > bin_y_low - delta_y
                # and obj_pos[2] < self.bin_pos[2] + self.bin_size[2]r
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
            if self.camera_depth:
                front_image, front_depth = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='frontview', depth=self.camera_depth)
                side_image, side_depth = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='sideview', depth=self.camera_depth)
                bird_image, bird_depth = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='birdview', depth=self.camera_depth)
                image = np.concatenate((front_image, side_image, bird_image), 1)
                depth = np.concatenate((front_depth, side_depth, bird_depth), 1)

                # norm
                depth_max = 0.99
                depth_min = 0.85
                depth = (depth - depth_min) / (depth_max - depth_min)
                depth = np.clip(depth, 0, 1)
                depth = np.uint8(depth * 255)

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)

                di["image"] = np.concatenate((image, depth), 2)
            else:
                front_image = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='frontview')
                side_image = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='sideview')
                bird_image = self.sim.render(width=self.camera_width, height=self.camera_height, camera_name='birdview')
                di["image"] = np.concatenate((front_image, side_image, bird_image), 1)

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