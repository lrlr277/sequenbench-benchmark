import os
IMAGE_DIR = {
"human": "/mnt/beegfs/xr/lm_multimodal/share/human_update",
"animal": "/mnt/beegfs/xr/lm_multimodal/share/split_image_update/animal"
}

def count_image(path):
    return len(os.listdir(path))

for key in IMAGE_DIR:
    types = os.listdir(IMAGE_DIR[key])
    type_cnts = {t: count_image(os.path.join(IMAGE_DIR[key], t)) for t in types}
    sorted_cnts = sorted(type_cnts.items(), key=lambda x: x[1], reverse=True)
    print(key)
    print(sorted_cnts)