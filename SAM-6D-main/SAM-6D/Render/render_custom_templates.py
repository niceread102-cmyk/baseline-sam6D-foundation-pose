import blenderproc as bproc

import os
import argparse
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
parser.add_argument('--mesh_color', nargs=3, type=int, metavar=('R', 'G', 'B'), default=None,
                    help="RGB color (0-255) to apply to untextured meshes (overrides --colorize/--base_color)")
parser.add_argument('--flat_shading', action='store_true',
                    help="Disable normal interpolation so flat faces render flat (useful for CAD/STL models)")
parser.add_argument('--roughness', type=float, default=None,
                    help="Principled BSDF roughness (0=mirror/glossy, 1=fully matte). Default: Blender's 0.5")
parser.add_argument('--metallic', type=float, default=None,
                    help="Principled BSDF metallic (0=plastic/dielectric, 1=metallic). Default: 0.0")
parser.add_argument('--light_energy', type=float, default=1000,
                    help="Point light energy in watts (default: 1000)")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    # Measure radius from the centroid so off-origin meshes scale correctly
    centroid = (model_points.max(axis=0) + model_points.min(axis=0)) / 2.0
    centered = model_points - centroid
    radius = float(np.max(np.linalg.norm(centered, axis=1)))

    return 1 / (2 * radius)


# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

# calculating the scale of CAD model
if args.normalize:
    scale = get_norm_info(args.cad_path)
else:
    scale = 1

for idx, cam_pose in enumerate(cam_poses):
    
    bproc.clean_up()

    # load object
    obj = bproc.loader.load_obj(args.cad_path)[0]
    obj.set_scale([scale, scale, scale])
    obj.set_cp("category_id", 1)

    # Center mesh centroid at world origin so the camera (looking at origin) sees it properly
    mesh_tri = trimesh.load(args.cad_path, force='mesh')
    centroid = mesh_tri.bounding_box.centroid
    obj.set_location([-scale * centroid[0], -scale * centroid[1], -scale * centroid[2]])

    # assigning material colors to untextured objects
    if args.mesh_color is not None:
        # Convert 0-255 RGB to 0.0-1.0 linear for Blender's Principled BSDF
        r, g, b = [c / 255.0 for c in args.mesh_color]
        material = bproc.material.create('obj')
        material.set_principled_shader_value('Base Color', [r, g, b, 1.0])
        obj.set_material(0, material)
    elif args.colorize:
        color = [args.base_color, args.base_color, args.base_color, 0.]
        material = bproc.material.create('obj')
        material.set_principled_shader_value('Base Color', color)
        obj.set_material(0, material)

    # apply roughness / metallic overrides to all materials on the object
    if args.roughness is not None or args.metallic is not None:
        for material in obj.get_materials():
            if args.roughness is not None:
                material.set_principled_shader_value('Roughness', args.roughness)
            if args.metallic is not None:
                material.set_principled_shader_value('Metallic', args.metallic)

    # disable smooth-shading so flat faces render flat (important for CAD/STL models)
    if args.flat_shading:
        for poly in obj.blender_obj.data.polygons:
            poly.use_smooth = False
        obj.blender_obj.data.update()

    # convert cnos camera poses to blender camera poses
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    bproc.camera.add_camera_pose(cam_pose)
    
    # set light
    light_scale = 2.5
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    light1.set_energy(args.light_energy)

    bproc.renderer.set_max_amount_of_samples(50)
    # render the whole pipeline
    data = bproc.renderer.render()
    # render nocs
    data.update(bproc.renderer.render_nocs())
    
    # check save folder
    save_fpath = os.path.join(args.output_dir, "templates")
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    # save rgb image
    color_bgr_0 = data["colors"][0]
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

    # save mask
    mask_0 = data["nocs"][0][..., -1]
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    
    # save nocs
    xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
    np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))