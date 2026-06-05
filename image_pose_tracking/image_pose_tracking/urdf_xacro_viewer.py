import argparse
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


# Edit these when running the script directly from an IDE.
# DEFAULT_MODEL can be "osim", "intrinsic", or "custom".
# If DEFAULT_MODEL is "custom", CUSTOM_XACRO_FILE must point to a .urdf.xacro file.
DEFAULT_MODEL = "osim"
CUSTOM_XACRO_FILE = "/home/yuxuan_cor/pose_ws/src/image_pose_tracking/config/right_arm_osim_shoulder.urdf.xacro"


@dataclass
class Geometry:
    kind: str
    size: Tuple[float, ...]
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Link:
    name: str
    visuals: List[Geometry] = field(default_factory=list)


@dataclass
class Mimic:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float = -math.pi
    upper: float = math.pi
    mimic: Optional[Mimic] = None


@dataclass
class Robot:
    name: str
    links: Dict[str, Link]
    joints: List[Joint]
    children_by_parent: Dict[str, List[Joint]]
    root_link: str


def parse_vector(value: Optional[str], default: Tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in value.split()], dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def transform_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def joint_transform(joint: Joint, value: float) -> np.ndarray:
    t = transform_matrix(joint.origin_xyz, joint.origin_rpy)
    if joint.joint_type in ("revolute", "continuous"):
        r = np.eye(4)
        r[:3, :3] = axis_angle_matrix(joint.axis, value)
        t = t @ r
    elif joint.joint_type == "prismatic":
        p = np.eye(4)
        p[:3, 3] = joint.axis * value
        t = t @ p
    return t


def expand_xacro(path: Path) -> ET.Element:
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=True) as urdf:
        result = subprocess.run(
            ["xacro", str(path), "-o", urdf.name],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return ET.parse(urdf.name).getroot()


def parse_robot(path: Path) -> Robot:
    root = expand_xacro(path)
    links: Dict[str, Link] = {}
    joints: List[Joint] = []

    for link_xml in root.findall("link"):
        name = link_xml.attrib["name"]
        visuals = []
        for visual_xml in link_xml.findall("visual"):
            origin_xml = visual_xml.find("origin")
            xyz = parse_vector(origin_xml.attrib.get("xyz") if origin_xml is not None else None, (0, 0, 0))
            rpy = parse_vector(origin_xml.attrib.get("rpy") if origin_xml is not None else None, (0, 0, 0))
            geometry_xml = visual_xml.find("geometry")
            if geometry_xml is None:
                continue
            if geometry_xml.find("cylinder") is not None:
                cylinder = geometry_xml.find("cylinder")
                visuals.append(Geometry("cylinder", (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])), xyz, rpy))
            elif geometry_xml.find("sphere") is not None:
                sphere = geometry_xml.find("sphere")
                visuals.append(Geometry("sphere", (float(sphere.attrib["radius"]),), xyz, rpy))
            elif geometry_xml.find("box") is not None:
                box = geometry_xml.find("box")
                visuals.append(Geometry("box", tuple(float(x) for x in box.attrib["size"].split()), xyz, rpy))
        links[name] = Link(name, visuals)

    for joint_xml in root.findall("joint"):
        origin_xml = joint_xml.find("origin")
        axis_xml = joint_xml.find("axis")
        limit_xml = joint_xml.find("limit")
        mimic_xml = joint_xml.find("mimic")
        lower, upper = -math.pi, math.pi
        if limit_xml is not None:
            lower = float(limit_xml.attrib.get("lower", lower))
            upper = float(limit_xml.attrib.get("upper", upper))
        mimic = None
        if mimic_xml is not None:
            mimic = Mimic(
                mimic_xml.attrib["joint"],
                float(mimic_xml.attrib.get("multiplier", 1.0)),
                float(mimic_xml.attrib.get("offset", 0.0)),
            )
        joints.append(
            Joint(
                name=joint_xml.attrib["name"],
                joint_type=joint_xml.attrib["type"],
                parent=joint_xml.find("parent").attrib["link"],
                child=joint_xml.find("child").attrib["link"],
                origin_xyz=parse_vector(origin_xml.attrib.get("xyz") if origin_xml is not None else None, (0, 0, 0)),
                origin_rpy=parse_vector(origin_xml.attrib.get("rpy") if origin_xml is not None else None, (0, 0, 0)),
                axis=parse_vector(axis_xml.attrib.get("xyz") if axis_xml is not None else None, (1, 0, 0)),
                lower=lower,
                upper=upper,
                mimic=mimic,
            )
        )

    children_by_parent: Dict[str, List[Joint]] = {}
    child_links = set()
    for joint in joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)
        child_links.add(joint.child)

    root_candidates = [name for name in links if name not in child_links]
    root_link = root_candidates[0] if root_candidates else next(iter(links))
    return Robot(root.attrib.get("name", path.stem), links, joints, children_by_parent, root_link)


def compute_link_transforms(robot: Robot, joint_values: Dict[str, float]) -> Dict[str, np.ndarray]:
    transforms = {robot.root_link: np.eye(4)}

    def value_for(joint: Joint) -> float:
        if joint.mimic:
            return joint.mimic.multiplier * joint_values.get(joint.mimic.joint, 0.0) + joint.mimic.offset
        return joint_values.get(joint.name, 0.0)

    def visit(link_name: str) -> None:
        parent_tf = transforms[link_name]
        for joint in robot.children_by_parent.get(link_name, []):
            transforms[joint.child] = parent_tf @ joint_transform(joint, value_for(joint))
            visit(joint.child)

    visit(robot.root_link)
    return transforms


def make_cylinder(radius: float, length: float) -> np.ndarray:
    theta = np.linspace(0, 2 * math.pi, 24)
    y = np.array([-length / 2.0, length / 2.0])
    theta_grid, y_grid = np.meshgrid(theta, y)
    x = radius * np.cos(theta_grid)
    z = radius * np.sin(theta_grid)
    return np.stack([x, y_grid, z], axis=-1).reshape(-1, 3)


def make_sphere(radius: float) -> np.ndarray:
    u = np.linspace(0, 2 * math.pi, 18)
    v = np.linspace(0, math.pi, 10)
    u_grid, v_grid = np.meshgrid(u, v)
    x = radius * np.cos(u_grid) * np.sin(v_grid)
    y = radius * np.sin(u_grid) * np.sin(v_grid)
    z = radius * np.cos(v_grid)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def make_box(size: Tuple[float, ...]) -> np.ndarray:
    sx, sy, sz = size
    return np.array(
        [
            [x * sx / 2.0, y * sy / 2.0, z * sz / 2.0]
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
    )


def transform_points(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    hom = np.c_[points, np.ones(len(points))]
    return (tf @ hom.T).T[:, :3]


def geometry_points(geometry: Geometry) -> np.ndarray:
    if geometry.kind == "cylinder":
        return make_cylinder(*geometry.size)
    if geometry.kind == "sphere":
        return make_sphere(*geometry.size)
    if geometry.kind == "box":
        return make_box(geometry.size)
    return np.empty((0, 3))


def draw_robot(ax, robot: Robot, joint_values: Dict[str, float]) -> None:
    ax.clear()
    transforms = compute_link_transforms(robot, joint_values)
    all_points = []

    for link in robot.links.values():
        link_tf = transforms.get(link.name)
        if link_tf is None:
            continue
        for visual in link.visuals:
            local_tf = transform_matrix(visual.origin_xyz, visual.origin_rpy)
            points = transform_points(geometry_points(visual), link_tf @ local_tf)
            if len(points) == 0:
                continue
            all_points.append(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, alpha=0.75)

    for joint in robot.joints:
        parent_tf = transforms.get(joint.parent)
        child_tf = transforms.get(joint.child)
        if parent_tf is None or child_tf is None:
            continue
        p0 = parent_tf[:3, 3]
        p1 = child_tf[:3, 3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="black", linewidth=1)

    points = np.vstack(all_points) if all_points else np.zeros((1, 3))
    center = points.mean(axis=0)
    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]), 0.25)
    half = span * 0.65
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(robot.name)
    ax.set_box_aspect((1, 1, 1))


def movable_joints(robot: Robot) -> List[Joint]:
    return [
        joint
        for joint in robot.joints
        if joint.joint_type in ("revolute", "continuous", "prismatic") and joint.mimic is None
    ]


def show_viewer(robot: Robot) -> None:
    joints = movable_joints(robot)
    joint_values = {joint.name: 0.0 for joint in joints}
    slider_height = max(0.18, min(0.55, 0.045 * len(joints) + 0.04))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.05, slider_height, 0.9, 0.95 - slider_height], projection="3d")

    sliders = []
    for index, joint in enumerate(joints):
        bottom = 0.03 + index * 0.04
        slider_ax = fig.add_axes([0.25, bottom, 0.65, 0.025])
        lower, upper = joint.lower, joint.upper
        if joint.joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        slider = Slider(slider_ax, joint.name, lower, upper, valinit=0.0)
        sliders.append(slider)

        def on_change(value, joint_name=joint.name):
            joint_values[joint_name] = value
            draw_robot(ax, robot, joint_values)
            fig.canvas.draw_idle()

        slider.on_changed(on_change)

    draw_robot(ax, robot, joint_values)
    plt.show()


def default_xacro_path(name: str) -> Path:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    if name == "custom":
        return Path(CUSTOM_XACRO_FILE).expanduser()
    if name == "osim":
        return config_dir / "right_arm_osim_shoulder.urdf.xacro"
    if name == "intrinsic":
        return config_dir / "right_arm.urdf.xacro"
    path = Path(name).expanduser()
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize a URDF/Xacro arm model with joint sliders.")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help="'osim', 'intrinsic', 'custom', or a path to a .urdf.xacro file",
    )
    parser.add_argument("--list-joints", action="store_true", help="print movable joints and exit")
    args = parser.parse_args(argv)

    xacro_path = default_xacro_path(args.model)
    if not xacro_path.exists():
        parser.error(f"model file does not exist: {xacro_path}")

    try:
        robot = parse_robot(xacro_path)
    except Exception as exc:
        print(f"Failed to load {xacro_path}: {exc}", file=sys.stderr)
        return 1

    if args.list_joints:
        for joint in movable_joints(robot):
            print(f"{joint.name}: [{joint.lower:.6g}, {joint.upper:.6g}] axis={joint.axis.tolist()}")
        return 0

    show_viewer(robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
