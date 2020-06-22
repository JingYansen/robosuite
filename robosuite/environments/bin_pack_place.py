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
from robosuite.scripts.utils import norm_depth

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from robosuite.models.arenas import BinPackingArena
from robosuite.models.objects import (
    BananaObject,
    BottleObject,
    BowlObject,
    BreadObject,
    CanObject,
    CerealObject,
    LemonObject,
    MilkObject,
    PlateWithHoleObject,
    RoundNutObject,
    SquareNutObject,
    BananaVisualObject,
    BottleVisualObject,
    BowlVisualObject,
    BreadVisualObject,
    CanVisualObject,
    CerealVisualObject,
    LemonVisualObject,
    MilkVisualObject,
    PlateWithHoleVisualObject,
    RoundNutVisualObject,
    SquareNutVisualObject,
)

from robosuite.models.tasks import BinPackingTask, UniformRandomSampler


class BinPackPlace(SawyerEnv, mujoco_env.MujocoEnv):
    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.39, 0.49, 0.82),
        table_target_size=(0.085, 0.085, 0.12),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
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
        camera_names=["birdview", "targetview"],
        camera_height=128,
        camera_width=128,
        camera_depth=True,
        camera_type='image+depth',
        keys='image',
        video_height=256,
        video_width=256,
        render_drop_freq=0,
        obj_names=['Milk'] * 1 + ['Bread'] * 1 + ['Cereal'] * 1 + ['Can'] * 1 + ['Banana'] * 1 + ['Bowl'] * 1 + ['Bottle'] * 1 + ['Lemon'] * 1,
        # obj_names=['Cereal'] * 6,
        force_ratios=0.2,
        z_limit=1.0,
        take_nums=6,
        random_take=False,
        use_typeVector=False,
        make_dataset=False,
        dataset_path='data/temp/',
        # action_bound=(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])),
        action_bound=(np.array([0.5, 0.3]), np.array([0.7, 0.5])),
        # action_bound=(np.array([0.53, 0.3]), np.array([0.67, 0.45])),
        # action_bound=(np.array([0.5575, 0.3375]), np.array([0.6425, 0.4225])),
    ):

        # task settings
        self.random_take = random_take
        self.obj_names = obj_names
        self.take_nums = take_nums
        self.camera_type = camera_type
        self.force_ratios = force_ratios
        self.z_limit = z_limit
        self.use_typeVector = use_typeVector
        self.make_dataset = make_dataset
        self.dataset_path = dataset_path
        self.dataset_count = 0

        if self.make_dataset:
            self.label_file = os.path.join(self.dataset_path, 'label.txt')
            if os.path.exists(self.label_file):
                os.remove(self.label_file)

        assert self.take_nums <= len(self.obj_names)

        self.success_objs = 0

        self.render_drop_freq = render_drop_freq
        self.video_height = video_height
        self.video_width = video_width
        self.camera_names = camera_names

        self.single_object_mode = single_object_mode
        self.object_to_id = {
            "Milk"          : 0,
            "Bread"         : 1,
            "Cereal"        : 2,
            "Can"           : 3,
            "Banana"        : 4,
            "Bowl"          : 5,
            "Lemon"         : 6,
            "Bottle"        : 7,
            "PlateWithHole" : 8,
            "RoundNut"      : 9,
            "SquaredNut"    : 10,
        }
        self.obj_to_take = -1

        # settings for table top
        self.table_full_size = table_full_size
        self.table_target_size = table_target_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

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

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
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

        if keys is None:
            assert self.use_object_obs, "Object observations need to be enabled."
            keys = ["image", "state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(super().reset(), verbose=True)
        self.obs_dim = flat_ob.shape
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        high = action_bound[1]
        low = action_bound[0]
        assert np.all(high >= low)
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

        if self.use_typeVector:
            ob_image = np.concatenate(ob_lst)
            type_vector = np.zeros(len(self.object_to_id))

            obs = np.concatenate((ob_image.reshape(-1), type_vector))
            return obs
        else:
            return np.concatenate(ob_lst)

    def _name2obj(self, name):
        assert name in self.object_to_id.keys()

        if name == 'Milk':
            return MilkObject, MilkVisualObject
        elif name == 'Bread':
            return BreadObject, BreadVisualObject
        elif name == 'Cereal':
            return CerealObject, CerealVisualObject
        elif name == 'Can':
            return CanObject, CanVisualObject
        elif name == 'Banana':
            return BananaObject, BananaVisualObject
        elif name == 'Bowl':
            return BowlObject, BowlVisualObject
        elif name == 'Lemon':
            return LemonObject, LemonVisualObject
        elif name == 'Bottle':
            return BottleObject, BottleVisualObject
        elif name == 'PlateWithHole':
            return PlateWithHoleObject, PlateWithHoleVisualObject
        elif name == 'RoundNut':
            return RoundNutObject, RoundNutVisualObject
        elif name == 'SquaredNut':
            return SquareNutObject, SquareNutVisualObject

    def _make_objects(self, names):
        self.item_names = []
        self.ob_inits = []
        self.vis_inits = []

        idx = [0] * len(self.object_to_id)

        for name in names:
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
        self.mujoco_arena = BinPackingArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.5, -0.3, 0])

        # make objects by names
        self._make_objects(self.obj_names)


        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]), self.vis_inits[j]()))
        self.visual_objects = lst

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]), ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = BinPackingTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
        )
        self.model.place_objects()
        # self.model.place_visual()

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
        self.teleport_object(obj, np.array([10, 10]))

    def teleport_object(self, obj, action):
        """
        Teleport an object to a certain position (x, y, z).
        """
        assert obj in self.mujoco_objects.keys()
        assert 2 <= len(action) <= 3

        x, y = action[0], action[1]

        if len(action) == 2:
            z = self.z_limit
        else:
            z = action[2]

        sim_state = self.sim.get_state()
        x_dim = self.sim.model.get_joint_qpos_addr(obj)[0]
        y_dim = x_dim + 1
        z_dim = x_dim + 2

        sim_state.qpos[x_dim] = x
        sim_state.qpos[y_dim] = y
        sim_state.qpos[z_dim] = z

        self.sim.set_state(sim_state)
        self.sim.forward()

    def obj2type(self, obj_name):
        type = -1
        for obj_type, val in self.object_to_id.items():
            if obj_type in obj_name:
                type = val
                break

        assert type >= 0
        return type

    def take_an_object(self, action):
        ## random take an object
        if self.random_take:
            obj_idxs = np.nonzero(self.objects_not_take)[0]
            if len(obj_idxs) is 0:
                print('Warning: All objects have been taken.')
                self.obj_to_take = 0
                raise ValueError('no object to take.')
            else:
                self.obj_to_take = np.random.choice(obj_idxs)
        ## fix order to take
        else:
            self.obj_to_take = (self.objects_not_take != 0).argmax(axis=0)

        assert self.objects_not_take[self.obj_to_take] == 1
        self.objects_not_take[self.obj_to_take] = 0

        obj = self.object_names[self.obj_to_take]
        self.target_object = obj
        self.teleport_object(obj, action)

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
        self.sim.data.qfrc_applied[beg_dim + 2] = self.sim.data.qfrc_bias[beg_dim + 2] * self.force_ratios
        # self.sim.data.qfrc_applied[beg_dim+2:beg_dim+6] = self.sim.data.qfrc_bias[beg_dim+2:beg_dim+6]

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

        if self.make_dataset:
            ob_dict = self._get_observation()
            data_input = ob_dict['image'].copy()

        # take an obj
        self.take_an_object(action)

        self.timestep += 1
        self._pre_action(action)
        end_time = self.cur_time + self.control_timestep

        info = {}

        if self.render_drop_freq:
            i = 0
            info['birdview'] = []

        while self.cur_time < end_time:
            if self.render_drop_freq:
                if i % self.render_drop_freq == 0:
                    info['birdview'].append(self.sim.render(width=self.video_width, height=self.video_height,
                                                            camera_name='birdview', depth=self.camera_depth))

                i += 1

            self.sim.step()
            self.cur_time += self.model_timestep

        reward = self.reward(action)
        if reward > 0:
            self.success_objs += 1

        ## done
        done = (np.sum(self.objects_not_take != 1) >= self.take_nums)

        if done:
            info['success_obj'] = self.success_objs
            print('Done!')

        ob_dict = self._get_observation()

        # make data
        if self.make_dataset:
            def arr2str(arr):
                s = ''
                for i, a in enumerate(arr):
                    s += str(a)
                    if i != len(arr) - 1:
                        s += ','
                return s

            data_position = action.copy()
            data_type = self.obj2type(self.target_object)
            data_reward = reward

            # image
            img_dir = os.path.join(self.dataset_path, str(data_type))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            img_path = os.path.join(img_dir, str(self.dataset_count) + '.npy')
            np.save(img_path, data_input)

            # info
            label = img_path + ' ' + arr2str(data_position) + ' ' + str(data_type) + ' ' + str(data_reward)

            with open(self.label_file, 'a+') as f:
                f.write(label + '\n')

            self.dataset_count += 1

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

        self.success_objs = 0
        # reset positions of objects, and move objects out of the scene depending on the mode
        self.model.place_objects()

    def reward(self, action=None):
        # compute sparse rewards
        # last_num = np.sum(self.objects_in_bins)
        # self._check_success()
        # reward = np.sum(self.objects_in_bins) - last_num
        succ = self._check_success_obj(self.target_object)
        if succ:
            reward = 1
        else:
            # if self.in_box_bound(action):
            #     reward = 0
            # else:
            #     reward = -0.1
            reward = 0

        # if reward != 0:
        print('Reward: ', reward, ' by Action: ', action)
        return reward

    def get_bin_bound(self):
        bin_x_low = self.bin_pos[0] - self.bin_size[0] / 2
        bin_y_low = self.bin_pos[1] - self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0]
        bin_y_high = bin_y_low + self.bin_size[1]

        return np.array([bin_x_low, bin_y_low]), np.array([bin_x_high, bin_y_high])

    def not_in_bin(self, obj_pos):

        bin_x_low = self.bin_pos[0] - self.bin_size[0] / 2
        bin_y_low = self.bin_pos[1] - self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0]
        bin_y_high = bin_y_low + self.bin_size[1]

        delta = 0.02

        res = True
        if (
            obj_pos[2] > self.bin_pos[2]
            and obj_pos[0] < bin_x_high
            and obj_pos[0] > bin_x_low
            and obj_pos[1] < bin_y_high
            and obj_pos[1] > bin_y_low
            and obj_pos[2] < self.bin_pos[2] + self.bin_size[2] - delta
        ):
            res = False
        return res

    def in_box_bound(self, action):

        bin_x_low = self.bin_pos[0] - self.bin_size[0] / 2
        bin_y_low = self.bin_pos[1] - self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0]
        bin_y_high = bin_y_low + self.bin_size[1]

        if (
                action[0] <= bin_x_high
                and action[0] >= bin_x_low
                and action[1] <= bin_y_high
                and action[1] >= bin_y_low
        ):
            return True
        else:
            return False


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
            bird_image, bird_depth = self.sim.render(width=self.camera_width, height=self.camera_height,
                                                     camera_name='birdview', depth=self.camera_depth)

            image = bird_image
            depth = bird_depth

            # norm
            depth = norm_depth(depth)

            imgae_depth = np.concatenate((image, depth), 2)

            if self.camera_type == 'image+depth':
                di["image"] = imgae_depth
            elif self.camera_type == 'image':
                di["image"] = image
            elif self.camera_type == 'depth':
                di["image"] = depth
            else:
                raise ValueError('No such camera type: ', self.camera_type)

            di['vis'] = imgae_depth

        ## get type one-hot vector
        # if self.random_take:
        #     temp_idx = np.zeros(len(self.object_to_id))
        #     if self.obj_to_take >= 0:
        #         obj_to_take_name = self.obj_names[self.obj_to_take]
        #         obj_type = self.object_to_id[obj_to_take_name]
        #         temp_idx[obj_type] = 1
        #
        #     di["obj_taken"] = np.copy(temp_idx)

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

    def _check_success_obj(self, obj_name):
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
        not_in_bin = self.not_in_bin(obj_pos)
        if not_in_bin:
            return False
        else:
            return True

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