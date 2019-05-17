import rawpy
import glob
import os
from PIL.ExifTags import TAGS
from PIL import Image
import imageio
from tqdm import tqdm

def get_exif(fn):
    ret={}
    i=Image.open(fn)
    info=i._getexif()
    for tag,value in info.items():
        decoded=TAGS.get(tag,tag)
        ret[decoded]=value
    return ret

img_root='/home/hda/nfs_disk/dataset/detection/'
raw_root=os.path.join(img_root,'raw')
rgb_root=os.path.join(img_root,'rgb')

fns=glob.glob(raw_root+'/'+'*.CR2')

for i in tqdm(range(len(fns))):
    raw_path=fns[i]
    _,fn=os.path.split(fns[i])
    rgb_path=os.path.join(rgb_root,fn[0:-3]+'JPG')

    ret=get_exif(rgb_path)

    iso=ret['ISOSpeedRatings']
    exptime=ret['ExposureTime']
    # print('iso',iso)
    # print('exptime',exptime)

    # exp=float(exptime[0])/exptime[1]

    ratio=min(exptime[1]/20+iso/200,8)
    # print('ratio',ratio)

    raw=rawpy.imread(raw_path)
    process_raw=raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8,exp_shift=8,bright=2)
    imageio.imsave('./lighten/'+fn[0:-3]+'jpg',process_raw )
    # break
