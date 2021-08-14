import os
import shutil
import glob
import random

classes = os.listdir('../../data/imagenet/train/')
new_classes = []
for item in classes:
    if os.path.isdir(os.path.join('../../data/imagenet/train/', item)):
        new_classes.append(item)
assert len(new_classes) == 1000

for name in new_classes:
    files = glob.glob("../../data/imagenet/train/" + name + "/*.JPEG")
    number = len(files)
    to_be_moved = random.sample(files, int(number * 0.125))

    for f in enumerate(to_be_moved, 1):
        # import ipdb; ipdb.set_trace()
        dest = os.path.join("../../data/imagenet_sampled/train/", name)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)

# classes = os.listdir('../../data/imagenet_sampled/train/')
# new_classes = []
# for item in classes:
#     if os.path.isdir(os.path.join('../../data/imagenet_sampled/train/', item)):
#         new_classes.append(item)
# assert len(new_classes) == 1000

for name in new_classes:
    files = glob.glob("../../data/imagenet_sampled/train/" + name + "/*.JPEG")
    number = len(files)
    to_be_moved = random.sample(files, int(number * 0.2))
    for f in enumerate(to_be_moved, 1):
        dest = os.path.join("../../data/imagenet_sampled/val/", name)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.move(f[1], dest)
