import os
from pytube import YouTube
from tqdm import tqdm
import urllib
import progressbar

import pandas as pd

class ProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_video(video_url, save_dir, filename=None):
    print(video_url)
    os.makedirs(save_dir, exist_ok=True)
    if "youtube.com" in video_url:
        yt = YouTube(video_url)
        yt = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        yt.download(save_dir)
        if filename is not None:
            save_path = os.path.join(save_dir, filename)
            os.rename(os.path.join(save_dir, yt.default_filename), save_path)
        else:
            save_path = os.path.join(save_dir, yt.default_filename)
    else:
        assert filename is not None
        save_path = os.path.join(save_dir, filename)
        if os.path.isfile(save_path):
            return save_path
        urllib.request.urlretrieve(video_url, save_path, ProgressBar())
    return save_path

def main(save_dir='/mnt/hdd/surgical_detection/data'):
    labels = pd.read_csv('data/videos_2.csv')
    ids = labels['id']
    links = labels['src']

    for i, link in tqdm(zip(ids, links)):
        if str(i) == 'nan' or str(link) == 'nan':
            continue
        try:
            download_video('https://'+str(link).lstrip('/'), save_dir, f'{i}.mp4')
        except Exception as e:
            print(f'\n\nCould not download: {i}. {link}')
            print(e)
            continue
        break
    exit()


    for i, video in tqdm(enumerate(vids)):
        download_video('http://dg1fmc8qbela5.cloudfront.net/video001.mp4', save_dir, f'video{i:03d}')
        try:
            download_video(video, save_dir, f'video{i:03d}')
        except Exception as e:
            print(f'\n\nCould not download: {i}. {video}')
            print(e)

if __name__=='__main__':
    main()
