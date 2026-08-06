"""Shared Unitree R1 configuration — the SAME articulation, observations and actions the walking
policy (r1_flat_v2 / model_3999 / policy.pt) was trained on. Import this ONLY after the Isaac Sim
app has been launched (it imports omni/isaaclab.sim modules).
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

R1_USD = r"C:\Users\aimat\Desktop\humanoid\imported_cad\unitree_r1.usd"

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
            ".*_shoulder_pitch_joint": 0.20, ".*_elbow_joint": 0.60,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "waist_yaw_joint", "waist_roll_joint"],
            effort_limit_sim=300,
            stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0, ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0, "waist_yaw_joint": 200.0, "waist_roll_joint": 200.0},
            damping={".*_hip_.*": 5.0, ".*_knee_joint": 5.0, "waist_.*": 5.0},
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=40, stiffness=20.0, damping=2.0, armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint", "head_pitch_joint", "head_yaw_joint"],
            effort_limit_sim=100, stiffness=40.0, damping=10.0, armature=0.01,
        ),
    },
)


@configclass
class R1FlatEnvCfg(G1FlatEnvCfg):
    """The R1 walking env — identical to what model_3999 / policy.pt was trained on."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
        )
        # anti-crawl (matches the trained policy's env)
        self.terminations.root_too_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={"target_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
        )
