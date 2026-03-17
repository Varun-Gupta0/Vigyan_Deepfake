import cv2
import numpy as np

def make_video_from_image(image_path, out_video_path, frames=30):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
        
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (w, h))
    
    for i in range(frames):
        # Add a tiny bit of random noise to each frame so it's not identical
        noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        out.write(noisy_img)
        
    out.release()
    print(f"Created {out_video_path}")

if __name__ == "__main__":
    make_video_from_image("real.jpg", "test_real.mp4")
    make_video_from_image("fake.jpg", "test_fake.mp4")
