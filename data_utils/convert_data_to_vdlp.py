import os
import sys

import cv2
from tqdm import tqdm

PHASE_LABELS_DIC = {
        'preparation': 0,
        'calottriangledissection': 1,
        'clippingcutting': 2,
        'gallbladderdissection': 3,
        'gallbladderpackaging': 4,
        'cleaningcoagulation': 5,
        'gallbladderretraction': 6
    }


def get_filename(f):
    r = os.path.basename(f)
    r = '.'.join(r.split('.')[:-1])
    return r

def crop_image(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    pre1_picture = image[left:left + width, bottom:bottom + height]  

    return pre1_picture

def process_image(image):
    #dim = (int(image.shape[1]/image.shape[0]*300), 300)
    #image = cv2.resize(image, dim)
    #image = crop_image(image)
    image = cv2.resize(image, (250,250))
    return image

def extract_frames(video_path, fps, save_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0
    while success:
        frame = process_image(frame)
        cv2.imwrite(os.path.join(save_dir, f'{count}.jpg'), frame)
        count += 1
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(1000*count*1/fps))
        success, frame = vidcap.read()
    return vidcap.get(cv2.CAP_PROP_FPS)

def get_labels(file_path):
    with open(file_path, 'r') as f:
        labels = f.readlines()

    frame_nums, phases = [], []
    for label in labels[1:]:
        frame_num, phase = label.split()
        frame_nums.append(int(frame_num))
        phases.append(PHASE_LABELS_DIC[phase.lower()])

    return frame_nums, phases

def write_labels_separate_lines(file_path, labels, video_folder):
    with open(file_path, 'a') as f:
        start = 0
        for i in range(1,len(labels)):
            if ((i == len(labels)-1)
                or labels[i] != labels[start]):
                # Transition
                f.write(f'{video_folder} {start:08d} {i-1:08d} {labels[start]}')
                f.write('\n')
                start = i

def write_labels(file_path, labels, video_folder):
    with open(file_path, 'a') as f:
        start = 0
        f.write(f'{video_folder} {start:08d} {len(labels)-1:08d}')
        for i in range(1,len(labels)):
            if labels[i] != labels[start]:
                # Transition
                f.write(f' {i-1:08d} {labels[start]}')
                start = i
            if i == len(labels) - 1:
                f.write(f' {i:08d} {labels[start]}')
        f.write('\n')

def main(data_dir, save_dir, fps):
    #for file in tqdm(sorted(list(os.listdir(os.path.join(data_dir, 'videos'))))):
    for file in tqdm(sorted(list(os.listdir(data_dir)))):
        if not file.endswith('.mp4'):
            continue
        fname = get_filename(file)

        # First requirement is to convert video to images
        folder = os.path.join(save_dir, 'videos', fname)
        os.makedirs(folder)
        video_fps = extract_frames(os.path.join(data_dir, file), fps, folder)

#        # Second requirement is to add to the annotations file
#        # Get labels
#        frame_nums, phase_labels = get_labels(os.path.join(data_dir, 'phase_annotations', fname+'-phase.txt'))
#        # Sample
#        #assert video_fps % fps == 0
#        rate = int(video_fps // fps)
#        frame_rate = frame_nums[::rate]
#        phase_labels = phase_labels[::rate]
#        # Write
#        write_labels(os.path.join(save_dir, 'annotations_single.txt'), phase_labels, os.path.join('videos/rgb',fname))


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
