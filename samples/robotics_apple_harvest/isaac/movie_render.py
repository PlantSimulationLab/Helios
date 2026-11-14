"""A side-by-side movie of episodes recorded by movie_record.py in Helios (MuJoCo) and IsaacSim (PhysX), or, without --isaac, a movie of
Helios's episodes alone, for versions of the task that IsaacSim does not have.

Both sides are drawn the same way, by the Helios visualizer, from each simulator's recorded poses: the tree with its real geometry, and the arm
as its collision shapes. Leaves are drawn for context; neither simulator simulates them (they ride rigidly on their wood, as in training). An inset
in each panel shows what the wrist camera of harvest_env_enable_camera() would see. The policies shown do not use it: they are given exact positions.

From the sample's build directory, with IsaacSim's Python (for numpy, Pillow and imageio-ffmpeg):
    C:/isaacsim/python.bat ../isaac/movie_render.py --helios helios_301.npz --isaac isaac_301.npz --output harvester.mp4 --still harvester.png
    C:/isaacsim/python.bat ../isaac/movie_render.py --helios helios_301.npz --output harvester_helios.mp4
"""

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from movie_record import as_floats, helios_library, helios_link_names

FPS = 25  # one frame per control step: real time
INSET_SCALE = 0.30  # the wrist camera's inset, as a fraction of the panel

# Camera, relative to the fruit being picked: beside the robot, a little above the fruit, looking across the arm at the tree
EYE_OFFSET = np.array([-1.1, -2.3, 0.45])
LOOK_OFFSET = np.array([-0.35, 0.0, 0.0])

OUTCOME_TEXT = {"success": ("Fruit detached", (46, 160, 67)), "collision": ("Protective stop", (230, 120, 20)), "timeout": ("Timeout", (200, 60, 60)), "slipped": ("Grip slipped", (200, 60, 60)), "knocked_off": ("Knocked off", (200, 60, 60))}


def quaternion_multiply(a, b):
    """Products of quaternions (..., 4) given as (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quaternion_conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quaternion_rotate(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4) given as (w, x, y, z)."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


class HeliosPoses:
    """Turns link poses recorded in a Helios environment, possibly with fewer links, into the link order of the environment that draws them."""

    def __init__(self, meta, names):
        index = {n: i for i, n in enumerate(names)}
        self.count = len(names)
        recorded = meta["link_names"]
        self.source = np.array([i for i, n in enumerate(recorded) if n in index])
        self.target = np.array([index[recorded[i]] for i in self.source])

    def __call__(self, poses):
        out = np.full((self.count, 7), np.nan, np.float32)
        out[self.target] = poses[self.source]
        return out


class IsaacPoses:
    """Turns IsaacSim body poses into Helios link poses: each link moves rigidly with the body that carries it, from the rest pose both start from."""

    def __init__(self, meta, names, rest):
        index = {n: i for i, n in enumerate(names)}
        body_index = {n: i for i, n in enumerate(meta["body_names"])}
        self.count = len(names)
        links, bodies, owners = [], [], []
        for name, (owner, _) in meta["owners"].items():
            if name in index and owner in body_index and owner in index:
                links.append(index[name])
                bodies.append(body_index[owner])
                owners.append(index[owner])
        self.links = np.array(links)
        self.bodies = np.array(bodies)
        rest = rest.astype(np.float64)
        self.link_rest = rest[self.links]
        self.owner_rest = rest[np.array(owners)]

    def __call__(self, bodies):
        frames = bodies[self.bodies].astype(np.float64)
        position = frames[:, :3]
        quaternion = frames[:, [6, 3, 4, 5]]  # (x, y, z, w) to (w, x, y, z)
        # The rotation of the body from its rest pose, applied to the link's rest pose
        delta = quaternion_multiply(quaternion, quaternion_conjugate(self.owner_rest[:, 3:]))
        out = np.full((self.count, 7), np.nan, np.float32)
        out[self.links, :3] = position + quaternion_rotate(delta, self.link_rest[:, :3] - self.owner_rest[:, :3])
        out[self.links, 3:] = quaternion_multiply(delta, self.link_rest[:, 3:])
        return out


def load(path):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    episodes = []
    for k, summary in enumerate(meta["episodes"]):
        episodes.append(dict(summary, poses=data[f"poses_{k}"], joints=data[f"joints_{k}"], gripped=data[f"gripped_{k}"], target=data[f"target_{k}"]))
    return meta, episodes


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def chip(draw, xy, text, color, size=26):
    f = font(size, True)
    box = draw.textbbox(xy, text, font=f)
    pad = 10
    draw.rounded_rectangle((box[0] - pad, box[1] - pad // 2 - 2, box[2] + pad, box[3] + pad // 2 + 2), radius=10, fill=color)
    draw.text(xy, text, font=f, fill=(255, 255, 255))


def label_panel(image, title, subtitle, episode_number, frame, episode, ended):
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle((0, 0, width, 78), fill=(20, 24, 32, 200))
    draw.text((18, 8), title, font=font(32, True), fill=(255, 255, 255))
    draw.text((18, 46), subtitle, font=font(20), fill=(205, 212, 225))
    draw.rectangle((0, height - 54, width, height), fill=(20, 24, 32, 170))
    seconds = min(frame, len(episode["poses"]) - 1) / FPS
    draw.text((18, height - 44), f"Episode {episode_number}    t = {seconds:4.2f} s", font=font(24), fill=(255, 255, 255))
    if ended:
        text, color = OUTCOME_TEXT[episode["outcome"]]
        chip(draw, (18, 100), text, color + (235,))
    elif episode["outcome"] == "success" and frame > episode.get("detach_step", len(episode["poses"])):
        chip(draw, (18, 100), "Detached, carrying it away", OUTCOME_TEXT["success"][1] + (225,), 22)
    else:
        gripped = bool(episode["gripped"][min(frame, len(episode["gripped"]) - 1)])
        chip(draw, (18, 100), "Gripping, pulling" if gripped else "Approaching", ((40, 110, 200) if gripped else (90, 96, 110)) + (225,), 22)
    return image


def add_inset(picture, wrist):
    """The wrist camera's view in the upper right, below the title bar, labelled: the action is lower down."""
    width, height = picture.size
    inset = wrist.resize((int(width * INSET_SCALE), int(height * INSET_SCALE)), Image.LANCZOS)
    x = width - inset.width - 12
    y = 78 + 12 + 27
    draw = ImageDraw.Draw(picture, "RGBA")
    draw.rectangle((x - 3, y - 27, x + inset.width + 2, y + inset.height + 2), fill=(20, 24, 32, 220))
    draw.text((x + 4, y - 24), "Wrist camera (not used)", font=font(16, True), fill=(255, 255, 255))
    picture.paste(inset, (x, y))


def card(size, lines):
    image = Image.new("RGB", size, (22, 26, 34))
    draw = ImageDraw.Draw(image)
    y = size[1] // 2 - sum(s + 18 for _, s, _ in lines) // 2
    for text, s, bold in lines:
        f = font(s, bold)
        w = draw.textlength(text, font=f)
        draw.text(((size[0] - w) / 2, y), text, font=f, fill=(255, 255, 255) if bold else (200, 208, 222))
        y += s + 18
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--helios", required=True)
    parser.add_argument("--isaac", default="", help="IsaacSim's recording, shown beside Helios's; without it, Helios's episodes fill the frame")
    parser.add_argument("--pairs", type=int, default=6, help="episodes shown from each simulator")
    parser.add_argument("--grip-line", default="The cup must touch the apple to grip it; after a detachment it backs away with the apple for 1 s, whatever the policy would do", help="the title card's line describing the grip")
    parser.add_argument("--closing", action="append", default=[], help="a line of the closing card (heading, results, footnote); repeat for each line, none for no card")
    parser.add_argument("--closing-file", default="", help="text file with the closing card's lines, one per line, instead of --closing; needed for lines with %% signs when running through IsaacSim's python.bat, which runs under cmd")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hold", type=float, default=1.2, help="seconds the end of each episode is held")
    parser.add_argument("--no-leaves", action="store_true")
    parser.add_argument("--neighbors", type=int, default=0, help="trees drawn on each side of the simulated one along the row; drawn only, not simulated")
    parser.add_argument("--trellis", action="store_true", help="draw a tall-spindle trellis (wires, stakes, line posts); drawn only, not simulated")
    parser.add_argument("--radiation", action="store_true", help="draw with radiation cameras (sunlight, spectra, smooth shading) instead of the visualizer")
    parser.add_argument("--antialiasing", type=int, default=16, help="rays per pixel of the radiation cameras")
    parser.add_argument("--periodic", type=int, default=0, help="with --radiation, repeat the drawn row (--neighbors, --trellis) this many times each way along the row with a periodic boundary; the robot is not repeated")
    parser.add_argument("--rows", type=int, default=0, help="with --radiation, repeat the row this many times on each side across the rows (3.5 m apart) with a periodic boundary")
    parser.add_argument("--output", default="harvester.mp4")
    parser.add_argument("--still", default="", help="PNG of a representative frame")
    parser.add_argument("--stills-only", action="store_true", help="write only the still, for choosing the camera")
    parser.add_argument("--eye-offset", default=",".join(str(v) for v in EYE_OFFSET), help="camera position relative to the fruit being picked, x,y,z in meters")
    parser.add_argument("--look-offset", default=",".join(str(v) for v in LOOK_OFFSET), help="point the camera looks at, relative to the fruit being picked, x,y,z in meters")
    args, _ = parser.parse_known_args()
    eye_offset = np.array([float(v) for v in args.eye_offset.split(",")])
    look_offset = np.array([float(v) for v in args.look_offset.split(",")])
    if eye_offset.shape != (3,) or look_offset.shape != (3,):
        raise ValueError("--eye-offset and --look-offset each take three numbers, x,y,z")
    if args.closing_file:
        if args.closing:
            raise ValueError("give the closing card's lines with --closing or --closing-file, not both")
        with open(args.closing_file, encoding="utf-8") as file:
            args.closing = [line.rstrip("\n") for line in file if line.strip()]

    helios_meta, helios_episodes = load(args.helios)
    side_by_side = bool(args.isaac)
    if side_by_side:
        isaac_meta, isaac_episodes = load(args.isaac)
        if helios_meta["tree"] != isaac_meta["tree"]:
            raise ValueError(f"the recordings are of different trees ({helios_meta['tree']} and {isaac_meta['tree']})")
    else:
        isaac_episodes = helios_episodes
    tree = helios_meta["tree"]

    lib = helios_library()
    panel = (args.width // 2, args.height) if side_by_side else (args.width, args.height)
    seeds = (ctypes.c_int * 1)(tree)
    env = lib.harvest_env_create_for_movie(seeds, 1, 1, 1, 0, 0 if args.no_leaves else 1, 1)
    if not env:
        raise RuntimeError("The Helios environment for drawing could not be created")
    if (args.neighbors or args.trellis) and not lib.harvest_env_add_movie_scenery(env, args.neighbors, 1 if args.trellis else 0):
        raise RuntimeError("The rest of the row could not be added")
    if args.radiation:
        wrist_size = (int(panel[0] * INSET_SCALE), int(panel[1] * INSET_SCALE))
        if not lib.harvest_env_enable_radiation_movie_cameras(env, panel[0], panel[1], wrist_size[0], wrist_size[1], args.antialiasing, args.periodic, args.rows):
            raise RuntimeError("The radiation cameras could not be set up")
        frame_directory = tempfile.mkdtemp(prefix="harvest_frames_")
    elif not lib.harvest_env_enable_movie_camera(env, panel[0], panel[1]):
        raise RuntimeError("The Helios environment for drawing could not be created")
    names = helios_link_names(lib, env)
    rest = np.zeros((len(names), 7), np.float32)
    lib.harvest_env_rest_link_poses(env, 0, as_floats(rest))
    to_helios = HeliosPoses(helios_meta, names)
    if side_by_side:
        to_isaac = IsaacPoses(isaac_meta, names, rest)
        print(f"{len(names)} links drawn; {len(to_helios.target)} posed from Helios, {len(to_isaac.links)} from IsaacSim, the rest carried by their parents", flush=True)
    else:
        print(f"{len(names)} links drawn; {len(to_helios.target)} posed from Helios, the rest carried by their parents", flush=True)

    image = np.zeros((panel[1], panel[0], 3), np.uint8)

    def render(poses, joints, target):
        """The view from outside, and the wrist camera's view of the same pose."""
        eye = (target + eye_offset).astype(np.float32)
        look = (target + look_offset).astype(np.float32)
        joints = np.ascontiguousarray(joints, np.float32)
        poses = np.ascontiguousarray(poses)
        if args.radiation:
            movie_file = ctypes.create_string_buffer(4096)
            wrist_file = ctypes.create_string_buffer(4096)
            if not lib.harvest_env_render_radiation_movie_frame(env, 0, as_floats(poses), as_floats(joints), as_floats(eye), as_floats(look), frame_directory.encode(), movie_file, wrist_file, 4096):
                raise RuntimeError("harvest_env_render_radiation_movie_frame failed")
            images = []
            for name in (movie_file.value.decode(), wrist_file.value.decode()):
                with Image.open(name) as picture:
                    images.append(picture.convert("RGB"))
                os.remove(name)
            return images[0], images[1]
        pixels = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        if not lib.harvest_env_render_movie_frame(env, 0, as_floats(poses), as_floats(joints), as_floats(eye), as_floats(look), pixels):
            raise RuntimeError("harvest_env_render_movie_frame failed")
        outside = Image.fromarray(image.copy())
        if not lib.harvest_env_render_wrist_frame(env, 0, as_floats(poses), as_floats(joints), pixels):
            raise RuntimeError("harvest_env_render_wrist_frame failed")
        return outside, Image.fromarray(image.copy())

    policy = helios_meta["policy"]
    scripted = policy == "scripted"
    trained_in = "IsaacSim" if "isaac" in policy else "Helios"
    controller = "the scripted controller" if scripted else f"a policy trained in {trained_in}"
    subtitle = f"Tree {tree}, " + ("scripted controller" if scripted else f"policy trained in {trained_in}") + "  ·  drawn by Helios"

    # The last view rendered for each side, by (episode, step): the frames that hold the end of an episode show the same pose, and rendering it again
    # costs as much as any frame with the radiation cameras
    last_rendered = {}

    def side(episodes, convert, title, k, frame):
        episode = episodes[k]
        last = len(episode["poses"]) - 1
        f = min(frame, last)
        cached = last_rendered.get(title)
        if cached is None or cached[0] != (k, f):
            cached = ((k, f),) + render(convert(episode["poses"][f]), episode["joints"][f], episode["target"].astype(np.float64))
            last_rendered[title] = cached
        picture, wrist = cached[1].copy(), cached[2]
        add_inset(picture, wrist)
        return label_panel(picture, title, subtitle, k + 1, frame, episode, frame >= last)

    def compose(k, frame):
        if not side_by_side:
            return side(helios_episodes, to_helios, "MuJoCo  (Helios)", k, frame)
        canvas = Image.new("RGB", (args.width, args.height))
        canvas.paste(side(helios_episodes, to_helios, "MuJoCo  (Helios)", k, frame), (0, 0))
        canvas.paste(side(isaac_episodes, to_isaac, "PhysX  (Isaac Sim)", k, frame), (panel[0], 0))
        ImageDraw.Draw(canvas).rectangle((panel[0] - 2, 0, panel[0] + 1, args.height), fill=(22, 26, 34))
        return canvas

    pairs = min(args.pairs, len(helios_episodes), len(isaac_episodes))
    if args.still or args.stills_only:
        # A frame with the cup on the fruit: the step before the end of the first episode
        k = 0
        frame = min(len(helios_episodes[k]["poses"]), len(isaac_episodes[k]["poses"])) - 6
        started = time.perf_counter()
        still = compose(k, frame)
        print(f"rendered the still in {time.perf_counter() - started:.1f} s (the first frame also builds the scene for the renderer)", flush=True)
        if args.radiation:
            started = time.perf_counter()
            compose(k, frame - 1)
            print(f"rendered the next frame in {time.perf_counter() - started:.1f} s", flush=True)
        still.save(args.still or "movie_still.png")
        print(f"wrote {args.still or 'movie_still.png'}", flush=True)
        if args.stills_only:
            return

    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(args.output, (args.width, args.height), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", output_params=["-crf", "21", "-preset", "slow", "-movflags", "+faststart"], macro_block_size=8)
    writer.send(None)
    frames = 0

    def emit(picture, repeat=1):
        nonlocal frames
        data = np.asarray(picture, np.uint8).tobytes()
        for _ in range(repeat):
            writer.send(data)
            frames += 1

    size = (args.width, args.height)
    drawn_only = ([] if args.no_leaves else ["leaves"]) + (["the rest of the row"] if args.periodic else ["the neighboring trees"] if args.neighbors else []) + (["the trellis"] if args.trellis else [])
    drawn_only_line = []
    if drawn_only:
        listed = drawn_only[0] if len(drawn_only) == 1 else ", ".join(drawn_only[:-1]) + " and " + drawn_only[-1]
        drawn_only_line = [(listed[0].upper() + listed[1:] + " are drawn for context; " + ("neither simulator simulates them" if side_by_side else "they are not simulated"), 20, False)]
    if side_by_side:
        title_lines = [("Apple harvesting robot, two physics engines", 46, True), (("The same tree and the same scripted controller" if scripted else f"The same tree and the same policy (trained in {trained_in})") + ", simulated in Helios (MuJoCo) and Isaac Sim (PhysX)", 24, False), ("Episodes are not paired: each simulator draws its own fruit, force to detach and start position", 22, False), ("Both sides " + ("rendered by Helios radiation cameras" if args.radiation else "drawn by the Helios visualizer") + " from each simulator's recorded poses  ·  real time", 20, False)] + drawn_only_line
    else:
        title_lines = [("Apple harvesting robot in Helios", 46, True), (f"Tree {tree}, never used in training, and {controller}, simulated in Helios (MuJoCo)", 24, False), ((("Rendered by Helios radiation cameras: sunlight, measured spectra, smooth shading") if args.radiation else "Drawn by the Helios visualizer") + "  ·  real time", 20, False)] + drawn_only_line
    title_lines += [(("The controller" if scripted else "The policy") + " is given exact positions of the fruit and wood, not images; the inset shows what a wrist camera would see", 20, False), (args.grip_line, 20, False)]
    emit(card(size, title_lines), int(4.0 * FPS))
    for k in range(pairs):
        length = max(len(helios_episodes[k]["poses"]), len(isaac_episodes[k]["poses"]))
        for frame in range(length + int(args.hold * FPS)):
            emit(compose(k, frame))
        print(f"episode {k + 1}/{pairs}: {helios_episodes[k]['outcome']}" + (f" / {isaac_episodes[k]['outcome']}" if side_by_side else ""), flush=True)
    if args.closing:
        emit(card(size, [(args.closing[0], 34, True)] + [(line, 26, False) for line in args.closing[1:-1]] + [(args.closing[-1], 22, False)]), int(4.5 * FPS))
    writer.close()
    lib.harvest_env_destroy(env)
    print(f"wrote {args.output}: {frames} frames, {frames / FPS:.1f} s", flush=True)


if __name__ == "__main__":
    main()
