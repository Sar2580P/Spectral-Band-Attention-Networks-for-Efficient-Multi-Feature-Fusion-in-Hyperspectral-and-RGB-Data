# import os
# import pandas as pd
# import json
# import cv2
# from tqdm import tqdm
# import numpy as np
# from spectral.io import envi


# class HyperspectralPCA(BaseModel):
#     def read_hdr(path):
#         data = envi.open(path+'.hdr', path)
#         return data

#     def split_image(image, x, y, h, w, num_rows, num_cols):
#         segment_height = h // num_rows
#         segment_width = w // num_cols
#         segments = []
#         try:
#             for row in range(num_rows):
#                 for col in range(num_cols):
#                     start_x = x + col * segment_width
#                     start_y = y + row * segment_height
#                     end_x = start_x + segment_width
#                     end_y = start_y + segment_height
#                     segment = image[start_y:end_y, start_x:end_x]

#                     thresholded_segment = segment[:, :, 50] > 0.19
#                     indices = np.argwhere(thresholded_segment)
#                     center_x = int(indices[:, 1].mean()) + start_x
#                     center_y = int(indices[:, 0].mean()) + start_y
#                     new_segment = image[center_y - 20 : center_y + 20, center_x - 12 : center_x + 12]
#                     segments.append(new_segment)
#         except:
#             pass
#         return segments

#     def create_cropping_jpg(img_path):
#         # Find seeds
#         image = cv2.imread(img_path)
#         image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

#         image = cv2.flip(image, 1)[400:-400, 1200:-900]

#         # Apply threshold to create a binary image
#         # ideal t = 161 using otsu
#         _, binary = cv2.threshold(image[:,:,2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#         # Find contours in the binary image
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Filter contours based on area
#         seeds, ar = [], []
        
#         for contour in contours:
#             a = cv2.contourArea(contour)
            
#             if a > 4000 and a<100000:
#                 seeds.append(contour)
#                 ar.append(a)
#         count = len(seeds)

#         if count!=72:
#             cv2.imwrite(os.path.join('Data/skipped', img_path.split('\\')[-1]), image)
#             return None, count

#         contour_images = []
#         for i, contour in enumerate(seeds):
#             # Perimeter and Area
#             x, y, w, h = cv2.boundingRect(contour)
#             w2, h2 = int(w / 2), int(h / 2)
#             centroid_x, centroid_y= x + w2, y + h2

#             if h>247 or w>120 :
#                 print("out of bounds ", w, ' ', h, ' ', img_path, ' ', i)
#             contour_image = image[centroid_y - 122: centroid_y + 125, centroid_x - 60: centroid_x + 60]
#             contour_images.append({'img': contour_image, 'x':centroid_x, 'y':centroid_y})

#         # Numbering
#         contour_images = sorted(contour_images, key=lambda obj: obj['y'], reverse=False)
#         contour_images = [sorted(contour_images[i:i+6], key=lambda obj: obj['x'], reverse = False) for i in range(0, 72, 6)]
#         contour_images_ = [element['img'] for group in contour_images for element in group]
  
#         return contour_images_, count

# def segment_images():
#   class_to_id = {}
#   id_to_class = {}
  
#   rootdir = "dataset_v1"
#   final_dir_hsi = "Data/hsi"
#   final_dir_rgb = "Data/rgb"
#   df_rgb, df_hsi = pd.DataFrame(columns=['path', 'class_id']), pd.DataFrame(columns=['path', 'class_id'])

#   skip_ct, ct  = 0 , 0
#   for dirpath, dirnames, _ in os.walk(rootdir):

#       for dirname in dirnames:
#           path = os.path.join(dirpath, dirname)
#           class_ = dirname
#           if class_ not in class_to_id:
#             class_to_id[class_] = ct
#             id_to_class[ct] = class_
#             ct += 1

#           for dirpath_, _, filenames in os.walk(path):
#             for file in filenames:
#               img_path = os.path.join(dirpath_, file)

#               if  file.lower().endswith(".jpg"):
#                 start=0
#                 if file[-5]=='2':
#                   start = 72
#                 contour_images, count = create_cropping_jpg(img_path)

#                 for i, img in enumerate(contour_images):
#                   cv2.imwrite(os.path.join(final_dir_rgb, '{ct}_{i}.png'.format(ct = ct-1 ,i=i+start)), img)
#                   df_rgb.loc[len(df_rgb)] = ['{ct}-{i}.png'.format(ct = ct-1 ,i=i+start), ct-1]

#               elif  file.lower().endswith(".bil"):
#                 start=0
#                 if file[-5]=='2':
#                   start = 72
#                 img = read_hdr(img_path)
#                 img = np.array(img.load())
#                 images = split_image(img, 25, 75, 700, 280, 12, 6)


#                 for i, seed_image in enumerate(images):
#                   name = '{}_{}.npy'.format(ct-1, start+i)
#                   try:
#                     np.save(os.path.join(final_dir_hsi, name), seed_image)
#                     df_hsi.loc[len(df_hsi)] = [name, ct-1]
#                   except:
#                     print("error in writing hsi images", file)
#                     skip_ct += 1


#   df_rgb.to_csv('Data/rgb.csv', index = False)
#   df_hsi.to_csv('Data/hsi.csv', index = False)
#   print("\n\nskipped ", skip_ct, " images")
#   mappings = {'class_to_id': class_to_id, 'id_to_class': id_to_class}

#   j = json.dumps(mappings, indent=4)
#   with open('Data/mappings.json', 'w') as f:
#       print(j, file=f)
      
# # segment_images()


import os
import pandas as pd
import json
import cv2
import numpy as np
from pydantic import BaseModel, DirectoryPath
from spectral.io import envi

class HyperspectralImageProcessor(BaseModel):
    root_dir: DirectoryPath
    final_dir_hsi: DirectoryPath
    final_dir_rgb: DirectoryPath
    csv_rgb_path: str
    csv_hsi_path: str
    json_mappings_path: str

    def read_hdr(self, path: str):
        data = envi.open(path + '.hdr', path)
        return data

    def split_image(self, image, x, y, h, w, num_rows, num_cols):
        segment_height = h // num_rows
        segment_width = w // num_cols
        segments = []
        try:
            for row in range(num_rows):
                for col in range(num_cols):
                    start_x = x + col * segment_width
                    start_y = y + row * segment_height
                    end_x = start_x + segment_width
                    end_y = start_y + segment_height
                    segment = image[start_y:end_y, start_x:end_x]

                    thresholded_segment = segment[:, :, 50] > 0.19
                    indices = np.argwhere(thresholded_segment)
                    if indices.size > 0:
                        center_x = int(indices[:, 1].mean()) + start_x
                        center_y = int(indices[:, 0].mean()) + start_y
                        new_segment = image[center_y - 20: center_y + 20, center_x - 12: center_x + 12]
                        segments.append(new_segment)
        except Exception as e:
            print(f"Error in split_image: {e}")
        return segments

    def create_cropping_jpg(self, img_path: str):
        # Find seeds
        image = cv2.imread(img_path)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)[400:-400, 1200:-900]

        # Apply threshold to create a binary image
        _, binary = cv2.threshold(image[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        seeds, ar = [], []
        
        for contour in contours:
            a = cv2.contourArea(contour)
            
            if 4000 < a < 100000:
                seeds.append(contour)
                ar.append(a)
        count = len(seeds)

        if count != 72:
            cv2.imwrite(os.path.join('Data/skipped', img_path.split('\\')[-1]), image)
            return None, count

        contour_images = []
        for i, contour in enumerate(seeds):
            # Perimeter and Area
            x, y, w, h = cv2.boundingRect(contour)
            w2, h2 = int(w / 2), int(h / 2)
            centroid_x, centroid_y = x + w2, y + h2

            if h > 247 or w > 120:
                print("out of bounds ", w, ' ', h, ' ', img_path, ' ', i)
            contour_image = image[centroid_y - 122: centroid_y + 125, centroid_x - 60: centroid_x + 60]
            contour_images.append({'img': contour_image, 'x': centroid_x, 'y': centroid_y})

        # Numbering
        contour_images = sorted(contour_images, key=lambda obj: obj['y'], reverse=False)
        contour_images = [sorted(contour_images[i:i + 6], key=lambda obj: obj['x'], reverse=False) for i in range(0, 72, 6)]
        contour_images_ = [element['img'] for group in contour_images for element in group]
  
        return contour_images_, count

    def segment_images(self):
        class_to_id = {}
        id_to_class = {}
        
        df_rgb, df_hsi = pd.DataFrame(columns=['path', 'class_id']), pd.DataFrame(columns=['path', 'class_id'])

        skip_ct, ct = 0, 0
        for dirpath, dirnames, _ in os.walk(self.root_dir):
            for dirname in dirnames:
                path = os.path.join(dirpath, dirname)
                class_ = dirname
                if class_ not in class_to_id:
                    class_to_id[class_] = ct
                    id_to_class[ct] = class_
                    ct += 1

                for dirpath_, _, filenames in os.walk(path):
                    for file in filenames:
                        img_path = os.path.join(dirpath_, file)

                        if file.lower().endswith(".jpg"):
                            start = 0
                            if file[-5] == '2':
                                start = 72
                            contour_images, count = self.create_cropping_jpg(img_path)

                            if contour_images is not None:
                                for i, img in enumerate(contour_images):
                                    cv2.imwrite(os.path.join(self.final_dir_rgb, '{ct}_{i}.png'.format(ct=ct-1, i=i+start)), img)
                                    df_rgb.loc[len(df_rgb)] = ['{ct}-{i}.png'.format(ct=ct-1, i=i+start), ct-1]

                        elif file.lower().endswith(".bil"):
                            start = 0
                            if file[-5] == '2':
                                start = 72
                            img = self.read_hdr(img_path)
                            img = np.array(img.load())
                            images = self.split_image(img, 25, 75, 700, 280, 12, 6)

                            for i, seed_image in enumerate(images):
                                name = '{}_{}.npy'.format(ct-1, start+i)
                                try:
                                    np.save(os.path.join(self.final_dir_hsi, name), seed_image)
                                    df_hsi.loc[len(df_hsi)] = [name, ct-1]
                                except Exception as e:
                                    print(f"Error in writing HSI images: {e}")
                                    skip_ct += 1

        df_rgb.to_csv(self.csv_rgb_path, index=False)
        df_hsi.to_csv(self.csv_hsi_path, index=False)
        print("\n\nskipped ", skip_ct, " images")
        mappings = {'class_to_id': class_to_id, 'id_to_class': id_to_class}

        j = json.dumps(mappings, indent=4)
        with open(self.json_mappings_path, 'w') as f:
            f.write(j)


if __name__=='__main__':
    processor = HyperspectralImageProcessor(
                            root_dir="dataset_v1",
                            final_dir_hsi="Data/hsi",
                            final_dir_rgb="Data/rgb",
                            csv_rgb_path="Data/rgb.csv",
                            csv_hsi_path="Data/hsi.csv",
                            json_mappings_path="Data/mappings.json"
                            )

    # Run the image segmentation process
    processor.segment_images()