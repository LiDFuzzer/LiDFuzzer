from mmdet3d.apis import LidarSeg3DInferencer

def run_inference(model, pcd_data_list, device=None,
                  out_dir='/home/semLidarFuzz/service/models/outputs', show=False, wait_time=-1,
                  no_save_vis=False, no_save_pred=False, print_result=False):
    """
    Run inference on a list of point cloud data files using a specified model.
    
    Parameters:cd 
    - model_config: Path to the model config file.
    - model_weights: Path to the model weights file.
    - pcd_data_list: List of paths to the point cloud data files.
    - device: Inference device.
    - out_dir: Output directory for predictions and visualizations.
    - show: Whether to show online visualization results.
    - wait_time: Wait time for online visualization. A negative value means wait indefinitely.
    - no_save_vis: Do not save visualization results.
    - no_save_pred: Do not save prediction results.
    - print_result: Whether to print the inference results.
    """
    # Initialize inferencer
    if model == "Cylinder3D":
        model_config = "/home/semLidarFuzz/mmdetection3d/configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
        model_weights = "/home/semLidarFuzz/mmdetection3d/checkpoints/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth"
    elif model == "SPVCNN":
        model_config = "/home/semLidarFuzz/mmdetection3d/configs/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
        model_weights = "/home/semLidarFuzz/mmdetection3d/checkpoints/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908-d68a68b7.pth"
    elif model == "MinkuNet":
        model_config = "/home/semLidarFuzz/mmdetection3d/configs/minkunet/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
        model_weights = "/home/semLidarFuzz/mmdetection3d/checkpoints/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth"

    print(model_config,model_weights)

    inferencer = LidarSeg3DInferencer(model=model_config, weights=model_weights, device=device)

    # Iterate through each point cloud file in the list
    call_args = {
        'inputs': {'points': pcd_data_list},
        'out_dir': out_dir,
        'show': show,
        'wait_time': wait_time,
        'no_save_vis': no_save_vis,
        'no_save_pred': no_save_pred,
        'print_result': print_result
    }
        
    # Run inference on the current point cloud file
    inferencer(**call_args)
    # print(pcd_data)
    # Optional: Log processing of each file
    # print(f'Processed {pcd_data_list}')

    # Log saving results, if applicable
    if out_dir != '' and not (no_save_vis and no_save_pred):
        print(f'Results have been saved at {out_dir}')


# Example usage
if __name__ == '__main__':
    run_inference(
        model = "Cylinder3D",
        pcd_data_list=
            '/home/semanticKITTI/dataset/sequences/04/velodyne'
        ,
        show=False
    )
