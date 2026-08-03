from .multiview import load_data as load_multiview

def get_data(data_name, args):
    scene_info = load_multiview(data_name, **vars(args))

    train_dataset = scene_info.train_cameras
    test_dataset = scene_info.test_cameras
    video_dataset = scene_info.video_cameras

    return scene_info, train_dataset, test_dataset, video_dataset