import argparse
import cv2
import numpy as np
import os

def get_random_crop(image, mask, crop_h, crop_w):
    assert len(image.shape) == 3

    max_w = image.shape[1] - crop_w
    max_h = image.shape[0] - crop_h

    x = np.random.randint(0, max_w)
    y = np.random.randint(0, max_h)

    crop_image = image[y: y + crop_h, x: x + crop_w, :]
    crop_mask = mask[y: y + crop_h, x: x + crop_w, :]

    return crop_image, crop_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drive", type=str, default="", help="path to drive dataset"
    )
    parser.add_argument(
        "--chasedb", type=str, default="", help="path to chasedb dataset"
    )
    parser.add_argument(
        "--hrf", type=str, default="", help="path to hrf dataset"
    )
    parser.add_argument(
        "--saved_path", type=str, default="final_dataset", help="path for saving dataset"
    )

    args = parser.parse_args()

    datasets = {'DRIVE': args.drive, 'ChaseDB': args.chase_db, 'HRF': args.hrf}

    SAVED_PATH = args.saved_path

    for dataset_name, dataset_path in datasets.items():
        if dataset_path == "": continue
        
        save_pth_images = f'{SAVED_PATH}/{dataset_name}/Images'
        save_pth_masks = f'{SAVED_PATH}/{dataset_name}/Labels'

        os.makedirs(save_pth_images, exist_ok=True)
        os.makedirs(save_pth_masks, exist_ok=True)

        IMAGE_PATH = f'{dataset_path}/Images'
        MASK_PATH = f'{dataset_path}/Labels'

        for image in os.listdir(IMAGE_PATH):
            print(dataset_name, image)
            image_pth = f'{IMAGE_PATH}/{image}'
            label_pth = f'{MASK_PATH}/{image}' if dataset_name != 'HRF' else f"{MASK_PATH}/{image.split('.')[0]}.tif"

            image_ = cv2.imread(image_pth)
            mask_ = cv2.imread(label_pth)
            for i in range(10):
                image_crop, mask_crop = get_random_crop(image_, mask_, 256, 256)
                cv2.imwrite(f"{save_pth_images}/{image.split('.')[0]}_{i}.png", image_crop)
                cv2.imwrite(f"{save_pth_masks}/{image.split('.')[0]}_{i}.png", mask_crop)

            if max(image_.shape[:2]) / min(image_.shape[:2]) < 1.5:
                image_full = cv2.resize(image_, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                mask_full = cv2.resize(mask_, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(f"{save_pth_images}/{image}", image_crop)
                cv2.imwrite(f"{save_pth_masks}/{image}", mask_crop)