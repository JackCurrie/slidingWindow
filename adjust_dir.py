# So that analyze_frame program I wrote.... Well it sucks. It is missing frames in some odd places
# and that makes it so that ffmpeg is having a hard time doing what it needs to. so I am going to
# count and sort all
#

import os, re

dir_name = './out_sliding_75stride_95thresh/'

files = os.listdir(dir_name)

files = [{'old_fn': file, 'i': int(re.findall(r'\d+', file)[0])} for file in files]
files = sorted(files, key=lambda k: k['i'])

for i, file in enumerate(files):
    file['new_fn'] = 'image' + str(i) +'.jpg'

for file in files:
    os.rename(dir_name + file['old_fn'], dir_name + file['new_fn'])
