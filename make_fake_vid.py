import cv2
import sys

def make_fake(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("Cannot open", in_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frames_written = 0
    while cap.isOpened() and frames_written < 60: # Limit to 60 frames max to speed up
        ret, frame = cap.read()
        if not ret:
            break
            
        # Corrupt the center region
        ch, cw = h // 2, w // 2
        offset = min(h, w) // 4
        
        region = frame[ch-offset:ch+offset, cw-offset:cw+offset]
        small = cv2.resize(region, (30, 30), interpolation=cv2.INTER_LINEAR)
        big = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        frame[ch-offset:ch+offset, cw-offset:cw+offset] = big
        out.write(frame)
        frames_written += 1
        
    cap.release()
    out.release()
    print("Done writing fake video!")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        make_fake(sys.argv[1], sys.argv[2])
