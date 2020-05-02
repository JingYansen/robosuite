import os

from robosuite.environments.base import make
from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

from robosuite.environments.bin_pack_place import BinPackPlace
from robosuite.environments.bin_squeeze import BinSqueeze
from robosuite.environments.bin_squeeze_multi import BinSqueezeMulti

__version__ = "0.3.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
