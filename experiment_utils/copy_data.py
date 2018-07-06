# this script extract 90 degree gei image from gei feature data provided by CASIA
import shutil
import os

motion_name = [
    'bg-01', 'bg-02', 'cl-01', 'cl-02', 'nm-01',
    'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06'
]

src_dir = 'Z:/DatasetB/GEI_CASIA_B/GEI_CASIA_B/gei/'
target_dir = 'Z:/DatasetB/GEI_CASIA_B/GEI_CASIA_B/gei90/'

for i in range(1, 125):
    stri = ""
    if (i < 10):
        stri = "00" + str(i)
    elif (i < 100):
        stri = "0" + str(i)
    else:
        stri = str(i)
    for motion in motion_name:
        src = src_dir + stri + '/' + motion + '/' + stri + '-' + motion + \
            '-090.png'
        dst_dir = target_dir + stri + '/'
        dst = target_dir + stri + '/' + stri + '-' + motion + \
            '-090.png'
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        print("scr: " + src)
        print("dst: " + dst)
        shutil.copyfile(src, dst)