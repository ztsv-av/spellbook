# def preprocessClassificationDataset(image_filename, preprocessing_dict, is_val):
    
#     '''
#     preprocesses the given image
#     '''

#     image = loadJPGToNumpy(image_filename, dir=TRAIN_FILES_PATH, image_type=tf.uint8)
#     image = tf.cast(image, dtype=tf.float32)

#     if preprocessing_dict['normalization'] is not None:
#         image = preprocessing_dict['normalization'](image)

#     if preprocessing_dict['to_color']:
#         if len(image.shape) == 2:
#             image = tf.expand_dims(image, -1)
#         image = tf.image.grayscale_to_rgb(image)
    
#     if preprocessing_dict['resize']:
#         image = tf.image.resize(image, size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
#     if preprocessing_dict['permutations'] is not None:
#         if not is_val:
#             image = classification_permutations(image, preprocessing_dict['permutations'])

#     label = getLabelFromFilename(image_filename)

#     return image, label