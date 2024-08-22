#!/usr/bin/env python3

import glob
import os, sys
from PIL import Image
from ase import io

# trajs = io.read("md_simulation.traj", ':')
# cnt = 0
# for atoms in trajs[::100]:
#     io.write(f"./md_folder/{cnt}.png", atoms)
#     cnt += 1
# print(cnt)

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    print(frames)
    frame_one = frames[0]
    frame_one.save("my_md.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

# sys.exit()
# if __name__ == "__main__":
make_gif("md_folder")

