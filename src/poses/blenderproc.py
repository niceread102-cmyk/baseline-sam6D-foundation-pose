import blenderproc  # must be the very first import — blenderproc enforces this
"""
BlenderProc template rendering script for CNOS.

Called automatically by CNOS._render_blenderproc():
    blenderproc run src/poses/blenderproc.py <cad_path> <poses_npy> <output_dir>

The script receives obj_poses with translations in mm (predefined sphere poses * 0.4).
It divides by 1000 to convert to metres before placing the camera, matching the
pyrender path in cnos.py which does obj_poses[:, :3, 3] /= 1000.
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import mathutils

import bpy

# Must match cnos.py DEFAULT_INTRINSIC / DEFAULT_IMG_SIZE
INTRINSIC = np.array([[604.7352,    0.,      430.31723],
                      [  0.,      604.959,  245.13461],
                      [  0.,        0.,       1.      ]])
IMG_W, IMG_H = 640, 480

# Flip Y and Z to convert from OpenCV camera convention to OpenGL/Blender convention
OPENCV_TO_OPENGL = np.array([[1.,  0.,  0., 0.],
                              [0., -1.,  0., 0.],
                              [0.,  0., -1., 0.],
                              [0.,  0.,  0., 1.]])


def parse_args():
    # Blender injects its own argv before '--'; user args follow the separator.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", type=Path)
    parser.add_argument("poses_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mesh-color", nargs=3, type=int, default=None,
                        metavar=("R", "G", "B"),
                        help="RGB color (0-255) to apply to the mesh")
    return parser.parse_args(argv)


def import_mesh(cad_path: Path):
    """Import a mesh file into the current Blender scene. Returns list of MESH objects."""
    suffix = cad_path.suffix.lower()
    before = {o.name for o in bpy.data.objects}

    if suffix == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(cad_path))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(cad_path))
    elif suffix == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=str(cad_path))
        except AttributeError:
            bpy.ops.import_mesh.stl(filepath=str(cad_path))
    elif suffix == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=str(cad_path))
        except AttributeError:
            bpy.ops.import_mesh.ply(filepath=str(cad_path))
    elif suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(cad_path))
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=str(cad_path))
    else:
        raise ValueError(f"Unsupported mesh format: {cad_path.suffix!r}")

    after = {o.name for o in bpy.data.objects}
    return [bpy.data.objects[n] for n in (after - before)
            if bpy.data.objects[n].type == "MESH"]


def center_and_scale(mesh_objs):
    """Move the combined bounding-box centroid to world origin; scale mm→m if needed."""
    corners = []
    for obj in mesh_objs:
        for c in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(c))

    lo = np.array([min(c[i] for c in corners) for i in range(3)])
    hi = np.array([max(c[i] for c in corners) for i in range(3)])
    centroid = (lo + hi) / 2.0
    extents = hi - lo
    # Match cnos.py: np.linalg.norm(mesh.extents * 2) > 100 detects mm-scale meshes.
    # Using just norm(extents) gives ~80 for this object (below 100), so scale is never
    # applied. Doubling matches the trimesh convention and correctly gives ~162 > 100.
    scale = 0.001 if np.linalg.norm(extents * 2) > 100.0 else 1.0  # mm → m

    for obj in mesh_objs:
        obj.scale = mathutils.Vector((scale, scale, scale))
        obj.location = obj.location - mathutils.Vector(centroid * scale)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def set_flat_shading(mesh_objs):
    """Disable smooth shading — equivalent to pyrender.Mesh.from_trimesh(mesh, smooth=False)."""
    for obj in mesh_objs:
        obj.data.polygons.foreach_set('use_smooth', [False] * len(obj.data.polygons))
        obj.data.update()


def apply_color(mesh_objs, rgb_255):
    """Replace all materials on mesh_objs with a flat Principled BSDF of the given RGB color."""
    r, g, b = [c / 255.0 for c in rgb_255]
    mat = bpy.data.materials.new(name="CnosColor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.0
    for obj in mesh_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def main():
    args = parse_args()
    cad_path = args.cad_path
    poses_path = args.poses_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # obj_poses: object in camera frame (OpenCV), translations in mm
    obj_poses = np.load(str(poses_path)).astype(float)
    obj_poses[:, :3, 3] /= 1000.0  # mm → m

    blenderproc.init()

    # Transparent background so non-object pixels have alpha=0
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    # Load mesh and center it at world origin
    mesh_objs = import_mesh(cad_path)
    if not mesh_objs:
        raise RuntimeError(f"No MESH objects loaded from {cad_path}")
    center_and_scale(mesh_objs)
    set_flat_shading(mesh_objs)

    if args.mesh_color is not None:
        apply_color(mesh_objs, args.mesh_color)

    # Lighting: low ambient world + sun whose direction varies per frame so the
    # descriptors are robust to lighting changes (seeded for reproducibility).
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 0.15

    sun = blenderproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(2.5)

    # Camera intrinsics (same as cnos.py DEFAULT_INTRINSIC)
    blenderproc.camera.set_intrinsics_from_K_matrix(INTRINSIC, IMG_W, IMG_H)

    rng = np.random.default_rng(42)  # fixed seed → deterministic renders

    # Each obj_pose places the object in camera space (OpenCV convention).
    # Invert → camera-to-world in OpenCV, then right-multiply OPENCV_TO_OPENGL to
    # convert the camera's local frame from OpenCV (X right, Y down, Z fwd) to
    # Blender/OpenGL (X right, Y up, Z back). Left-multiplying would rotate the
    # world axes instead and puts the camera on the wrong side of the object.
    for frame_idx, obj_pose in enumerate(obj_poses):
        cam2world = np.linalg.inv(obj_pose) @ OPENCV_TO_OPENGL
        blenderproc.camera.add_camera_pose(cam2world)
        el = float(rng.uniform(0, np.pi))
        az = float(rng.uniform(0, 2 * np.pi))
        sun.set_rotation_euler([el, 0.0, az], frame=frame_idx)

    blenderproc.renderer.set_max_amount_of_samples(32)
    data = blenderproc.renderer.render()

    # Save RGBA PNGs named 000000.png … NNNNNN.png
    for idx, img in enumerate(data["colors"]):
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # film_transparent didn't produce alpha — synthesise from non-black pixels
            alpha = ((arr > 0).any(axis=2).astype(np.uint8) * 255)
            arr = np.dstack([arr, alpha])
        Image.fromarray(arr).save(output_dir / f"{idx:06d}.png")


if __name__ == "__main__":
    main()
