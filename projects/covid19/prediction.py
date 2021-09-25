import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.activations import sigmoid

import numpy as np
import cv2
import pandas as pd
import os
import pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm.notebook import tqdm

# import module for reading and updating configuration files.
from object_detection.utils import config_util

# import module for building the detection model
from object_detection.builders import model_builder

test_dir = 'data/classification/test/'
study_dirs = os.listdir(test_dir)

models_dir = 'trained_models/binaryCrossentropy/256_new/'
models = os.listdir(models_dir)
predictions_dir = 'predictions/binaryCrossentropy/256_new/'

def buildDetectionModel(num_classes = 1, checkpoint_path = 'object_detection_models/effdet0/all100/checkpoint/ckpt-0', dummy_shape=(1, 512, 512, 3)):

    # define the path to the .config file for model
    pipeline_config = 'object_detection_models/effdet0/all100/pipeline.config'

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config, config_override=None)

    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']

    # Modify the number of classes from its default
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    # model_config.ssd.freeze_batchnorm = True

    detection_model = model_builder.build(model_config=model_config, is_training=False)

    tmp_box_predictor_checkpoint = tf.train.Checkpoint(_base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads, _box_prediction_head=detection_model._box_predictor._box_prediction_head)

    tmp_model_checkpoint = tf.train.Checkpoint(_feature_extractor=detection_model._feature_extractor, _box_predictor=tmp_box_predictor_checkpoint)

    # Define a checkpoint that sets `model= None
    checkpoint = tf.train.Checkpoint(model = tmp_model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(checkpoint_path).expect_partial()

    # Run a dummy image through the model so that variables are created
    # For the dummy image, you can declare a tensor of zeros that has a shape that the preprocess() method can accept (i.e. [batch, height, width, channels]).

    # use the detection model's `preprocess()` method and pass a dummy image
    dummy = tf.zeros(shape=dummy_shape)
    tmp_image, tmp_shapes = detection_model.preprocess(dummy)

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)

    print('Weights restored!')

    return detection_model

def visualize(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray')
    
    plt.show()

def dicom2array(dcm_name, dcm_dir, voi_lut=True, fix_monochrome=True):
    
    dcm_file = pydicom.dcmread(dcm_dir + dcm_name)
    
    # to transform DICOM data to more simpler data
    if voi_lut:
        img_np = apply_voi_lut(dcm_file.pixel_array, dcm_file)
    else:
        img_np = dcm_file.pixel_array
    
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dcm_file.PhotometricInterpretation == "MONOCHROME1":
        img_np = np.amax(img_np) - img_np
    
    img_np = np.float32(img_np)
    
    return img_np

def ratio_convert(image):
    
    height = image.shape[0]
    width = image.shape[1]

    if height < width:
        diff2 = (width - height) // 2
        residual = (width - 2 * diff2) - height
        image = image[:, diff2: width - diff2 - residual]
        
    elif width < height:
        diff2 = (height - width) // 2
        residual = (height - 2 * diff2) - width
        image = image[diff2: height - diff2 - residual,:]
     
    return image

def resize_image(image, h, w):

    # create resize transform pipeline
    transformed = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_LANCZOS4)

    return transformed

def minMaxNormalize(x):
    try:
        x -= x.min()
        x /= x.max()

        return x

    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")

def addColorChannels(x, num_channels):

    try:
        x = np.repeat(x[..., np.newaxis], num_channels, -1)
        
        return x
    
    except AttributeError:
        raise TypeError("Convert input image 'x' to numpy array.")

def testPreprocessing(dcm_name, dcm_path, classification):

    # convert dcm to numpy
    image_inital = dicom2array(dcm_name, dcm_path)
    # make image as square
    # image = ratio_convert(image_inital)
    # resize image
    if classification:
        image = resize_image(image_inital, 256, 256)
        image = minMaxNormalize(image)
    else:
        image = resize_image(image_inital, 512, 512)
        image = (np.maximum(image,0) / image.max()) * 255.

    # add 3 channels
    image = addColorChannels(image, 3)

    return image, image_inital.shape

def save2np(file, numpy_dir, image):
    
    np_path = numpy_dir + file
    np.save(np_path, image)

def labelDecoder(idx):

    label = ''

    if idx == 0:
        label = 'negative'
    elif idx == 1:
        label = 'typical'
    elif idx == 2:
        label = 'indeterminate'
    else:
        label = 'atypical'

    return label

def detect(input_tensor, detection_model):
    """Run detection on an input image.

    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
    """
    preprocessed_input, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_input, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    
    return detections

detection_model = buildDetectionModel()

# def singleModelPrediction():

#     for modelFile in tqdm(models, desc='Models Progress'):

#         loadedModel = load_model(models_dir + modelFile, compile=False)

#         submission = {'id': [], 'PredictionString': []}
#         submission_image = {'id': [], 'PredictionString': []}

#         for study in tqdm(study_dirs, desc='Single Model Progess', leave=False):

#             study_path = test_dir + study + '/'

#             substudies = os.listdir(study_path)
#             prediction_total = np.zeros(4)
#             num_images = 0
#             for substudy in substudies:
#                 files_path = study_path + substudy + '/'

#                 files = os.listdir(files_path)
#                 for file in files:
                    
#                     # classification
#                     image_classification = testPreprocessing(file, files_path, classification=True)
#                     images = []
#                     images.append(image_classification)
#                     images = np.asarray(images)

#                     prediction = loadedModel.predict(images, batch_size=1)[0]
#                     prediction = sigmoid(prediction)
#                     prediction = prediction.numpy()

#                     prediction_total += prediction

#                     # localization
#                     image_detection = testPreprocessing(file, files_path, classification=False)
#                     image_detection = np.expand_dims(image_detection, axis=0)
#                     image_tensor = tf.convert_to_tensor(image_detection, dtype=tf.float32)
#                     detections = detect(image_tensor, detection_model=detection_model)
#                     boxes = detections['detection_boxes'][0].numpy()
#                     scores = detections['detection_scores'][0].numpy()
                    
#                     formatted_boxes = []
#                     boxes_idxs = np.nonzero(scores > 0.5)
#                     for idx in boxes_idxs:
#                         correct_box = np.zeros(4)
#                         x, y, width, height = 0, 0, 0, 0
#                         box = boxes[idx]
#                         for i, coordinate in enumerate(box):
#                             if i == 0:
#                                 y = coordinate * 512
#                                 correct_box[1] = y
#                             elif i == 1:
#                                 x = coordinate * 512
#                                 correct_box[0] = x
#                             elif i == 2:
#                                 width = coordinate * 512 - x
#                                 correct_box[2] = width
#                             else:
#                                 height = coordinate * 512 - y
#                                 correct_box[3] = height
#                         formatted_boxes.append(correct_box)

#                     if len(formatted_boxes) > 0:
#                         predictionIdImage = file.replace('.dcm', '_image')
#                         predictionStringImage = ''
#                         for i, box in enumerate(formatted_boxes):
#                             predictionString = 'opacity' + ' ' +  scores[boxes_idxs[i]] + ' ' + box[0] + ' ' + box[1] + ' ' + box[2] + ' ' + box[3]
#                             if i == 0:
#                                 predictionStringImage += predictionString
#                             else:
#                                 predictionStringImage += ' ' + predictionString
#                         submission_image['id'].append(predictionIdImage)
#                         submission_image['PredictionString'].append(predictionStringImage)

#                     if len(formatted_boxes) == 0:
#                         prediction_highestIdx = np.argmax(prediction)
#                         prediction_highest = prediction[prediction_highestIdx]
#                         if prediction_highest == 1:

#                             predictionIdImage = file.replace('.dcm', '_image')
#                             predictionStringImage = 'none' + ' ' +  '1' + ' ' + '0 0 1 1'
#                             submission_image['id'].append(predictionIdImage)
#                             submission_image['PredictionString'].append(predictionStringImage)

#                         else:
#                             predictionIdImage = file.replace('.dcm', '_image')
#                             predictionStringImage = 'none' + ' ' +  str(round(prediction_highest, 5)) + ' ' + '0 0 1 1'
#                             submission_image['id'].append(predictionIdImage)
#                             submission_image['PredictionString'].append(predictionStringImage)

#                     num_images += 1 

#             prediction_total /= num_images
            
#             prediction_row = []
            
#             for idx, confidence in enumerate(prediction_total):

#                 label = labelDecoder(idx)
#                 prediction_instance = ''

#                 if idx == 3:
#                     if confidence < 0.0001:
#                         prediction_instance = label + ' ' +  '0.0001' + ' ' + '0 0 1 1'
#                     elif confidence == 1:
#                         prediction_instance = label + ' ' +  '1' + ' ' + '0 0 1 1'
#                     else:    
#                         prediction_instance = label + ' ' +  str(round(confidence, 5)) + ' ' + '0 0 1 1'
#                 else:
#                     if confidence < 0.0001:
#                         prediction_instance = label + ' ' + '0.0001' + ' ' + '0 0 1 1 '
#                     elif confidence == 1:
#                         prediction_instance = label + ' ' +  '1' + ' ' + '0 0 1 1 '
#                     else:
#                         prediction_instance = label + ' ' + str(round(confidence, 5)) + ' ' + '0 0 1 1 '

#                 prediction_row.append(prediction_instance)
            
#             # if len(prediction_row) == 0 or max(prediction_total) < 0.1:
#             #     prediction_str = 'negative' + ' ' + str(round(random.uniform(0.8, 0.99999), 5)) + ' ' + '0 0 1 1'
#             #     predictionIdStudy = study + '_study'
#             #     predictionStringStudy = ','.join(prediction_str).replace(',', '')
#             #     submission['id'].append(predictionIdStudy)
#             #     submission['PredictionString'].append(predictionStringStudy)
#             #     print('CHECK ! ! !')
#             #     print(predictionIdStudy)

#             # else:
#             predictionIdStudy = study + '_study'
#             predictionStringStudy = ','.join(prediction_row).replace(',', '')
#             submission['id'].append(predictionIdStudy)
#             submission['PredictionString'].append(predictionStringStudy)

#         submission_df = pd.DataFrame(data=submission)
#         submission_df_image = pd.DataFrame(data=submission_image)
#         submission = submission_df.append(submission_df_image)
#         submission.to_csv(predictions_dir + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)

def ensemblePrediction():

    # load top models
    DenseNet121 = load_model(models_dir + 'DenseNet121_epoch-100.h5', compile=False)
    EffNet0 = load_model(models_dir + 'EfficientNetB0_epoch-100.h5', compile=False)
    EffNet1 = load_model(models_dir + 'EfficientNetB1_epoch-100.h5', compile=False)
    EffNet2 = load_model(models_dir + 'EfficientNetB2_epoch-100.h5', compile=False)
    EffNet3 = load_model(models_dir + 'EfficientNetB3_epoch-100.h5', compile=False)
    # EffNet4 = load_model(models_dir + 'EfficientNetB4_408.h5', compile=False)
    # InceptionResNetV2 = load_model(models_dir + 'InceptionResNetV2_391.h5', compile=False)
    # InceptionV3 = load_model(models_dir + 'InceptionV3_389.h5', compile=False)

    models = [DenseNet121, EffNet0, EffNet1, EffNet2, EffNet3] #, EffNet4, InceptionResNetV2, InceptionV3]

    submission = {'id': [], 'PredictionString': []}
    submission_image = {'id': [], 'PredictionString': []}

    # s = 0

    for study in tqdm(study_dirs, desc='Ensemble Progess'):

        study_path = test_dir + study + '/'
        substudies = os.listdir(study_path)

        prediction_total = np.zeros(4)
        num_images = 0

        for substudy in substudies:

            files_path = study_path + substudy + '/'

            files = os.listdir(files_path)

            for file in files:
                
                image, _ = testPreprocessing(file, files_path, classification=True)
                images = []
                images.append(image)
                images = np.asarray(images)

                ensemble_prediction = np.zeros(4)

                for model in models:

                    prediction = model.predict(images, batch_size=1)[0]
                    prediction = sigmoid(prediction)
                    prediction = prediction.numpy()
                    ensemble_prediction += prediction

                ensemble_prediction /= len(models)

                prediction_highestIdx = np.argmax(ensemble_prediction)
                prediction_highest = ensemble_prediction[prediction_highestIdx]
                prediction_highest = np.round(prediction_highest, 5)

                # localization
                if prediction_highestIdx != 0:

                    image_detection, initial_shape = testPreprocessing(file, files_path, classification=False)
                    image_detection = np.expand_dims(image_detection, axis=0)
                    image_tensor = tf.convert_to_tensor(image_detection, dtype=tf.float32)

                    detections = detect(image_tensor, detection_model=detection_model)
                    boxes = detections['detection_boxes'][0].numpy()
                    scores = detections['detection_scores'][0].numpy()

                    scores = np.round(scores, 5)
                    
                    #[ymin, xmin, ymax, xmax]
                    #[xmin, ymin, width, height]

                    boxes_idxs = np.nonzero(scores > 0.1)
                    boxes_ = boxes[boxes_idxs]
                    
                    if len(boxes_) > 0:

                        # boxes_[:, 0], boxes_[:, 2] = boxes_[:, 0] * initial_shape[0], boxes_[:, 2] * initial_shape[0]
                        # boxes_[:, 1], boxes_[:, 3] = boxes_[:, 1] * initial_shape[1], boxes_[:, 3] * initial_shape[1]

                        formatted_boxes = boxes_[:, [1, 0, 3, 2]]
                        formatted_boxes[:, 0], formatted_boxes[:, 2] = formatted_boxes[:, 0] * initial_shape[1], formatted_boxes[:, 2] * initial_shape[1]
                        formatted_boxes[:, 1], formatted_boxes[:, 3] = formatted_boxes[:, 1] * initial_shape[0], formatted_boxes[:, 3] * initial_shape[0]
                        formatted_boxes = np.round(formatted_boxes, 5)

                        new_boxes = []
                        for box in formatted_boxes:

                            diff_width = box[2] / initial_shape[1]
                            diff_height = box[3] / initial_shape[0]

                            if (diff_width > 0.6) or (diff_height > 0.6):
                                continue
                            else:
                                new_boxes.append(box)
                        
                        if len(new_boxes) > 0:

                            predictionIdImage = file.replace('.dcm', '_image')
                            predictionStringImage = ''

                            # num_boxes = 0
                            for i, box in enumerate(new_boxes):

                                # if (box[0] == 0) or (box[1] == 0):
                                #     continue
                                
                                predictionString = 'opacity' + ' ' +  str(scores[i]) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3])

                                if i == 0:
                                    predictionStringImage += predictionString
                                else:
                                    predictionStringImage += ' ' + predictionString
                                
                                # num_boxes += 1
                                # if num_boxes == 3:
                                #     break

                            submission_image['id'].append(predictionIdImage)
                            submission_image['PredictionString'].append(predictionStringImage)

                        else:

                            predictionIdImage = file.replace('.dcm', '_image')
                            predictionStringImage = 'none' + ' ' +  str(prediction_highest) + ' ' + '0 0 1 1'
                            submission_image['id'].append(predictionIdImage)
                            submission_image['PredictionString'].append(predictionStringImage)

                    else:

                        predictionIdImage = file.replace('.dcm', '_image')
                        predictionStringImage = 'none' + ' ' +  str(prediction_highest) + ' ' + '0 0 1 1'
                        submission_image['id'].append(predictionIdImage)
                        submission_image['PredictionString'].append(predictionStringImage)
                else:

                    predictionIdImage = file.replace('.dcm', '_image')
                    predictionStringImage = 'none' + ' ' +  str(prediction_highest) + ' ' + '0 0 1 1'
                    submission_image['id'].append(predictionIdImage)
                    submission_image['PredictionString'].append(predictionStringImage)

                prediction_total += ensemble_prediction
                num_images += 1 

        prediction_total /= num_images
        prediction_total = np.round(prediction_total, 5)
        
        prediction_row = []
        
        for idx, confidence in enumerate(prediction_total):

            label = labelDecoder(idx)
            prediction_instance = ''

            if idx == 3:
                prediction_instance = label + ' ' +  str(confidence) + ' ' + '0 0 1 1'
            else:
                prediction_instance = label + ' ' + str(confidence) + ' ' + '0 0 1 1 '

            prediction_row.append(prediction_instance)
        
        predictionIdStudy = study + '_study'
        predictionStringStudy = ','.join(prediction_row).replace(',', '')
        submission['id'].append(predictionIdStudy)
        submission['PredictionString'].append(predictionStringStudy)
        
        # s += 1
        # print('FINISHED ' + str(s) + '/' + str(len(study_dirs)))

    submission_df = pd.DataFrame(data=submission)
    submission_df_image = pd.DataFrame(data=submission_image)
    submission = submission_df.append(submission_df_image)
    submission.to_csv(predictions_dir + 'new_ensemble_box_notallboxes_0.1_' + '.csv', index=False)
    # print('FINISHED')

ensemblePrediction()

# FOR ALL CONFIDENCES ! ! !
    #     prediction_row = []
        
    #     for idx, confidence in enumerate(prediction_total):

    #         if confidence < 0.00001:
    #             continue

    #         label = labelDecoder(idx)
    #         prediction_instance = ''
            
    #         if confidence == 1:
    #             confidence = int(confidence)

    #         if idx == 3:
    #             prediction_instance = label + ' ' +  str(round(confidence, 5)) + ' ' + '0 0 1 1'
    #         else:
    #             prediction_instance = label + ' ' + str(round(confidence, 5)) + ' ' + '0 0 1 1 '
            
    #         prediction_row.append(prediction_instance)
        
    #     if len(prediction_row) == 0 or max(prediction_total) < 0.1:
    #         prediction_str = 'negative' + ' ' + str(round(random.uniform(0.8, 0.99999), 5)) + ' ' + '0 0 1 1'
    #         predictionIdStudy = study + '_study'
    #         predictionStringStudy = ','.join(prediction_str).replace(',', '')
    #         submission['id'].append(predictionIdStudy)
    #         submission['PredictionString'].append(predictionStringStudy)
    #         print('CHECK ! ! !')
    #         print(predictionIdStudy)

    #     else:
    #         predictionIdStudy = study + '_study'
    #         predictionStringStudy = ','.join(prediction_row).replace(',', '')
    #         submission['id'].append(predictionIdStudy)
    #         submission['PredictionString'].append(predictionStringStudy)

    # # np.save(predictions_dir + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(predictions_dir + 'submission_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
    # print('FINISHED ' + modelFile)


# FOR CONFIDENCE HIGHER THAN 0.25 ! ! !
    #     prediction_row = []
        
    #     for idx, confidence in enumerate(prediction_total):
            
    #         # if want only predictions > 0.5 in PredictionString
    #         if confidence < 0.25:
    #             continue

    #         label = labelDecoder(idx)
    #         prediction_instance = ''
            
    #         if confidence == 1:
    #             confidence = int(confidence)

    #         if idx == 3:
    #             prediction_instance = label + ' ' +  str(round(confidence, 5)) + ' ' + '0 0 1 1'
    #         else:
    #             prediction_instance = label + ' ' + str(round(confidence, 5)) + ' ' + '0 0 1 1 '
            
    #         prediction_row.append(prediction_instance)

    #     # check if all predictions < 0.25
    #     if len(prediction_row) == 0:
    #         highestIdx = np.argmax(prediction_total)
    #         highestValue = 0.25
    #         label = labelDecoder(highestIdx)
    #         prediction_str = label + ' ' +  str(highestValue) + ' ' + '0 0 1 1'

    #         predictionIdStudy = study + '_study'
    #         predictionStringStudy = ','.join(prediction_str).replace(',', '')
    #         submission['id'].append(predictionIdStudy)
    #         submission['PredictionString'].append(predictionStringStudy)
    #         print('CHECK ! ! !')
    #         print(predictionIdStudy)

    #     else:
    #         predictionIdStudy = study + '_study'
    #         predictionStringStudy = ','.join(prediction_row).replace(',', '')
    #         submission['id'].append(predictionIdStudy)
    #         submission['PredictionString'].append(predictionStringStudy)

    # # np.save(predictions_dir + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(predictions_dir + 'submission_higherFifty_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
    # print('FINISHED ' + modelFile)


# FOR HIGHEST CONFIDENCE ONLY ! ! !
    #     highest_idx = np.argmax(prediction_total)
    #     highest_confidence = prediction_total[highest_idx]
    #     if highest_confidence < 0.5:
    #         highest_confidence = 0.5
    
    #     label = labelDecoder(highest_idx)
    #
    #     prediction_str = label + ' ' + str(round(highest_confidence, 5)) + ' ' + '0 0 1 1'
    #     predictionIdStudy = study + '_study'
    #     predictionStringStudy = ','.join(prediction_str).replace(',', '')
    #     submission['id'].append(predictionIdStudy)
    #     submission['PredictionString'].append(predictionStringStudy)

    # # np.save(predictions_dir + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(predictions_dir + 'submission_onlyHighest_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
    # print('FINISHED ' + modelFile)