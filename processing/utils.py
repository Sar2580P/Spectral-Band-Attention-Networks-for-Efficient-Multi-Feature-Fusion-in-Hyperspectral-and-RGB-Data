import yaml

def read_yaml(CONFIG_PATH):
  with open(CONFIG_PATH, 'r') as f:
      config = yaml.safe_load(f)
  return config

#_______________________________________________________________________________________________________________________

## Splitting image logic
# import matplotlib.pyplot as plt


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


  


  

      