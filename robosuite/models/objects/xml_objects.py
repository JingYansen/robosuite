from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion


class BananaObject(MujocoXMLObject):
    """
    Banana object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/banana.xml"))


class BananaVisualObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion("objects/banana-visual.xml"))


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle.xml"))


class BottleVisualObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle-visual.xml"))


class BowlObject(MujocoXMLObject):
    """
    Bowl object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bowl.xml"))


class BowlVisualObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion("objects/bowl-visual.xml"))


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread-visual.xml"))


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can-visual.xml"))


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can.xml"))


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"))


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal.xml"))


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/lemon.xml"))


class LemonVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/lemon-visual.xml"))


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk.xml"))


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in SawyerPickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk-visual.xml"))


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in BaxterPegInHole)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"))


class PlateWithHoleVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in SawyerPickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/plate-with-hole-visual.xml"))


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round-nut.xml"))


class RoundNutVisualObject(MujocoXMLObject):
    """
    Round nut (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round-nut-visual.xml"))


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/square-nut.xml"))


class SquareNutVisualObject(MujocoXMLObject):
    """
    Square nut object (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/square-nut-visual.xml"))
