# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from estimater import *
from datareader import *
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/output_frames')
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    output_dir = args.output_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    os.makedirs(output_dir, exist_ok=True)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    
    result = NestDict()
    all_score_result = NestDict()
    all_pose_result = NestDict()
    gt_poses = NestDict()    
    
    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        video_id = reader.get_video_id()
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        gt_pose = reader.get_gt_pose_custom(i)
        if i==0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            ## for track_one, we need another function which perturbs the pose from the previous input keeping
            # the first pose where it is, perturb the other 9 poses with very small random perturbation or a fixed
            # perturbation in a geodesic dome with small angular and translational errors from the point of the 
            # previous pose estimate. Then, pass all the perturbed poses through a forward pass and generate
            # 10 pose hypotheses from the given 9 perturbed poses and 1 previous pose. 

            # Pass that through the constraints and the bound assumption / purse algorithm to generate a pose
            # uncertainty set. The abstract Purse formulation could then be used to generate the PurseReg set using
            # the closure algorithm. The closure algorithm then gives us a set of rotation and translation with a
            # confidence interval that is given by the user as the input. 
            mask = reader.get_mask(i).astype(bool)
            pose, perturbed_poses, perturbed_scores = est.track_uq(rgb=color, depth=depth, mask=mask, K=reader.K, iteration=args.track_refine_iter)            
            
            

            #pose = est.track_one(rgb=color, depth=depth, mask=mask, K=reader.K, iteration=args.track_refine_iter)
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt  ', pose.reshape(4,4))
        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            # Save the frame instead of displaying it
            cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.png'), vis[...,::-1])
        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
