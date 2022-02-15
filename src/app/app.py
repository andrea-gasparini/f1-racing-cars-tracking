import os

from utils import split_video_to_frames, merge_frames_to_video, generate_bounding_boxes

if __name__ == '__main__':
    app_path = os.path.join('src', 'application')
    video_path = os.path.join(app_path, 'test_video_tracking_f1.mp4')
    output_frames_path = os.path.join(app_path, 'out_frames')
    model_path = os.path.join(app_path, 'model.ckpt')
    
    
    #split_video_to_frames(video_path, output_frames_path)
    #merge_frames_to_video(output_frames_path, app_path, fps=60)
    generate_bounding_boxes(model_path, output_frames_path)
    
    