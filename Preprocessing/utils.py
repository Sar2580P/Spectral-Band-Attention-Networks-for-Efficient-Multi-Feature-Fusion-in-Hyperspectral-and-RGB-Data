import cv2
import os
import numpy as np
import yaml

def create_cropping_jpg(img_path):
  # Find seeds
  image = cv2.imread(img_path)
  image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

  image = cv2.flip(image, 1)[400:-400, 1200:-900]

  # Apply threshold to create a binary image
  # ideal t = 161 using otsu
  _, binary = cv2.threshold(image[:,:,2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

   # Find contours in the binary image
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours based on area
  seeds, ar = [], []
  
  for contour in contours:
    a = cv2.contourArea(contour)
    
    if a > 4000 and a<100000:
      seeds.append(contour)
      ar.append(a)
  count = len(seeds)

  if count!=72:
    cv2.imwrite(os.path.join('Data/skipped', img_path.split('\\')[-1]), image)
    return None, count

  contour_images = []
  for i, contour in enumerate(seeds):
    # Perimeter and Area
    x, y, w, h = cv2.boundingRect(contour)
    w2, h2 = int(w / 2), int(h / 2)
    centroid_x, centroid_y= x + w2, y + h2

    if h>247 or w>120 :
      print("out of bounds ", w, ' ', h, ' ', img_path, ' ', i)
    contour_image = image[centroid_y - 122: centroid_y + 125, centroid_x - 60: centroid_x + 60]
    contour_images.append({'img': contour_image, 'x':centroid_x, 'y':centroid_y})

  # Numbering
  contour_images = sorted(contour_images, key=lambda obj: obj['y'], reverse=False)
  contour_images = [sorted(contour_images[i:i+6], key=lambda obj: obj['x'], reverse = False) for i in range(0, 72, 6)]
  contour_images_ = [element['img'] for group in contour_images for element in group]
  return contour_images_, count

#_______________________________________________________________________________________________________________________

## Splitting image logic
from spectral.io import envi as envi
import matplotlib.pyplot as plt
def read_hdr(path):
  data = envi.open(path+'.hdr', path)
  return data

def split_image(image, x, y, h, w, num_rows, num_cols):
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
              center_x = int(indices[:, 1].mean()) + start_x
              center_y = int(indices[:, 0].mean()) + start_y
              new_segment = image[center_y - 20 : center_y + 20, center_x - 12 : center_x + 12]
              segments.append(new_segment)
    except:
      pass
    return segments

# path_hsi = 'dataset_v1\A9-30-1-21_22_K\CD_1.bil'
# path_rgb = 'dataset_v1\DBW222_K_22\DBW_222_K_22_CD2.JPG'
# contour_images, count = create_cropping_jpg(path_rgb)
# for i, img in enumerate(contour_images_):
#   cv2.imwrite('rgb_{i}.png'.format(i=i), img)

# eg = read_hdr(path_hsi)
# ex = np.array(eg.load())
# images = split_image(ex, 25, 75, 700, 280, 12, 6)
# # plt.imshow(ex[:,:,50])
# # plt.savefig('hsi.png')



# fig, axes = plt.subplots(12, 12, figsize=(15, 25))

# for i in range(144):
#   r,c = i//12, i%12
#   if c>=6:
#     j = i - 6*(r+1)
#     title = 'rgb-{j}'.format(j=j) 
#     axes[r, c].imshow(contour_images[j])
#     axes[r, c].set_title(title)
#   else :
#     j = i - 6*r
#     title = 'hsi-{j}'.format(j=j)
#     axes[r, c].imshow(images[j][:,:,50])
#     axes[r, c].set_title(title)
#   axes[r, c].axis('off')

# plt.savefig('rgb_hsi.png')
                        
#_______________________________________________________________________________________________________

def load_config(CONFIG_PATH):
  with open(CONFIG_PATH, 'r') as f:
      config = yaml.safe_load(f)
  return config
  


  

      