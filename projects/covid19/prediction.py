import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import sigmoid

import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm

from dataPreprocessing import testPreprocessing, labelDecoder

from globalVariables import (NUM_CLASSES_DETECTION, CHECKPOINT_PATH, CONFIG_PATH, DUMMY_SHAPE_DETECTION)
from models import buildDetectionModel



TEST_DIR = 'data/classification/test/'
STUDY_DIRS = os.listdir(TEST_DIR)

MODELS_DIR = 'trained_MODELS/binaryCrossentropy/256_new/'
MODELS = os.listdir(MODELS_DIR)
PREDICTIONS_DIR = 'predictions/binaryCrossentropy/256_new/'


detection_model = buildDetectionModel(NUM_CLASSES_DETECTION, CHECKPOINT_PATH, CONFIG_PATH, DUMMY_SHAPE_DETECTION)


def localizeBoxes(input_tensor, detection_model):
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


def singleModelPrediction():

    for modelFile in tqdm(MODELS, desc='MODELS Progress'):

        loadedModel = load_model(MODELS_DIR + modelFile, compile=False)

        submission = {'id': [], 'PredictionString': []}
        submission_image = {'id': [], 'PredictionString': []}

        for study in tqdm(STUDY_DIRS, desc='Single Model Progess', leave=False):

            study_path = TEST_DIR + study + '/'

            substudies = os.listdir(study_path)
            prediction_total = np.zeros(4)
            num_images = 0
            for substudy in substudies:
                files_path = study_path + substudy + '/'

                files = os.listdir(files_path)
                for file in files:

                    # classification
                    image_classification = testPreprocessing(file, files_path, classification=True)
                    images = []
                    images.append(image_classification)
                    images = np.asarray(images)

                    prediction = loadedModel.predict(images, batch_size=1)[0]
                    prediction = sigmoid(prediction)
                    prediction = prediction.numpy()

                    prediction_total += prediction

                    # localization
                    image_detection = testPreprocessing(file, files_path, classification=False)
                    image_detection = np.expand_dims(image_detection, axis=0)
                    image_tensor = tf.convert_to_tensor(image_detection, dtype=tf.float32)
                    detections = localizeBoxes(image_tensor, detection_model=detection_model)
                    boxes = detections['detection_boxes'][0].numpy()
                    scores = detections['detection_scores'][0].numpy()

                    formatted_boxes = []
                    boxes_idxs = np.nonzero(scores > 0.5)
                    for idx in boxes_idxs:
                        correct_box = np.zeros(4)
                        x, y, width, height = 0, 0, 0, 0
                        box = boxes[idx]
                        for i, coordinate in enumerate(box):
                            if i == 0:
                                y = coordinate * 512
                                correct_box[1] = y
                            elif i == 1:
                                x = coordinate * 512
                                correct_box[0] = x
                            elif i == 2:
                                width = coordinate * 512 - x
                                correct_box[2] = width
                            else:
                                height = coordinate * 512 - y
                                correct_box[3] = height
                        formatted_boxes.append(correct_box)

                    if len(formatted_boxes) > 0:
                        predictionIdImage = file.replace('.dcm', '_image')
                        predictionStringImage = ''
                        for i, box in enumerate(formatted_boxes):
                            predictionString = 'opacity' + ' ' +  scores[boxes_idxs[i]] + ' ' + box[0] + ' ' + box[1] + ' ' + box[2] + ' ' + box[3]
                            if i == 0:
                                predictionStringImage += predictionString
                            else:
                                predictionStringImage += ' ' + predictionString
                        submission_image['id'].append(predictionIdImage)
                        submission_image['PredictionString'].append(predictionStringImage)

                    if len(formatted_boxes) == 0:
                        prediction_highestIdx = np.argmax(prediction)
                        prediction_highest = prediction[prediction_highestIdx]
                        if prediction_highest == 1:

                            predictionIdImage = file.replace('.dcm', '_image')
                            predictionStringImage = 'none' + ' ' +  '1' + ' ' + '0 0 1 1'
                            submission_image['id'].append(predictionIdImage)
                            submission_image['PredictionString'].append(predictionStringImage)

                        else:
                            predictionIdImage = file.replace('.dcm', '_image')
                            predictionStringImage = 'none' + ' ' +  str(round(prediction_highest, 5)) + ' ' + '0 0 1 1'
                            submission_image['id'].append(predictionIdImage)
                            submission_image['PredictionString'].append(predictionStringImage)

                    num_images += 1

            prediction_total /= num_images

            prediction_row = []

            for idx, confidence in enumerate(prediction_total):

                label = labelDecoder(idx)
                prediction_instance = ''

                if idx == 3:
                    if confidence < 0.0001:
                        prediction_instance = label + ' ' +  '0.0001' + ' ' + '0 0 1 1'
                    elif confidence == 1:
                        prediction_instance = label + ' ' +  '1' + ' ' + '0 0 1 1'
                    else:
                        prediction_instance = label + ' ' +  str(round(confidence, 5)) + ' ' + '0 0 1 1'
                else:
                    if confidence < 0.0001:
                        prediction_instance = label + ' ' + '0.0001' + ' ' + '0 0 1 1 '
                    elif confidence == 1:
                        prediction_instance = label + ' ' +  '1' + ' ' + '0 0 1 1 '
                    else:
                        prediction_instance = label + ' ' + str(round(confidence, 5)) + ' ' + '0 0 1 1 '

                prediction_row.append(prediction_instance)

            # if len(prediction_row) == 0 or max(prediction_total) < 0.1:
            #     prediction_str = 'negative' + ' ' + str(round(random.uniform(0.8, 0.99999), 5)) + ' ' + '0 0 1 1'
            #     predictionIdStudy = study + '_study'
            #     predictionStringStudy = ','.join(prediction_str).replace(',', '')
            #     submission['id'].append(predictionIdStudy)
            #     submission['PredictionString'].append(predictionStringStudy)
            #     print('CHECK ! ! !')
            #     print(predictionIdStudy)

            # else:
            predictionIdStudy = study + '_study'
            predictionStringStudy = ','.join(prediction_row).replace(',', '')
            submission['id'].append(predictionIdStudy)
            submission['PredictionString'].append(predictionStringStudy)

        submission_df = pd.DataFrame(data=submission)
        submission_df_image = pd.DataFrame(data=submission_image)
        submission = submission_df.append(submission_df_image)
        submission.to_csv(PREDICTIONS_DIR + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)

def ensemblePrediction():

    # load top MODELS
    DenseNet121 = load_model(MODELS_DIR + 'DenseNet121_epoch-100.h5', compile=False)
    EffNet0 = load_model(MODELS_DIR + 'EfficientNetB0_epoch-100.h5', compile=False)
    EffNet1 = load_model(MODELS_DIR + 'EfficientNetB1_epoch-100.h5', compile=False)
    EffNet2 = load_model(MODELS_DIR + 'EfficientNetB2_epoch-100.h5', compile=False)
    EffNet3 = load_model(MODELS_DIR + 'EfficientNetB3_epoch-100.h5', compile=False)
    # EffNet4 = load_model(MODELS_DIR + 'EfficientNetB4_408.h5', compile=False)
    # InceptionResNetV2 = load_model(MODELS_DIR + 'InceptionResNetV2_391.h5', compile=False)
    # InceptionV3 = load_model(MODELS_DIR + 'InceptionV3_389.h5', compile=False)

    MODELS = [DenseNet121, EffNet0, EffNet1, EffNet2, EffNet3] #, EffNet4, InceptionResNetV2, InceptionV3]

    submission = {'id': [], 'PredictionString': []}
    submission_image = {'id': [], 'PredictionString': []}

    # s = 0

    for study in tqdm(STUDY_DIRS, desc='Ensemble Progess'):

        study_path = TEST_DIR + study + '/'
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

                for model in MODELS:

                    prediction = model.predict(images, batch_size=1)[0]
                    prediction = sigmoid(prediction)
                    prediction = prediction.numpy()
                    ensemble_prediction += prediction

                ensemble_prediction /= len(MODELS)

                prediction_highestIdx = np.argmax(ensemble_prediction)
                prediction_highest = ensemble_prediction[prediction_highestIdx]
                prediction_highest = np.round(prediction_highest, 5)

                # localization
                if prediction_highestIdx != 0:

                    image_detection, initial_shape = testPreprocessing(file, files_path, classification=False)
                    image_detection = np.expand_dims(image_detection, axis=0)
                    image_tensor = tf.convert_to_tensor(image_detection, dtype=tf.float32)

                    detections = localizeBoxes(image_tensor, detection_model=detection_model)
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
        # print('FINISHED ' + str(s) + '/' + str(len(STUDY_DIRS)))

    submission_df = pd.DataFrame(data=submission)
    submission_df_image = pd.DataFrame(data=submission_image)
    submission = submission_df.append(submission_df_image)
    submission.to_csv(PREDICTIONS_DIR + 'new_ensemble_box_notallboxes_0.1_' + '.csv', index=False)
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

    # # np.save(PREDICTIONS_DIR + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(PREDICTIONS_DIR + 'submission_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
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

    # # np.save(PREDICTIONS_DIR + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(PREDICTIONS_DIR + 'submission_higherFifty_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
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

    # # np.save(PREDICTIONS_DIR + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', ''), submission)
    # submission_df = pd.DataFrame(data=submission)
    # submission_df_image = pd.DataFrame(data=submission_image)
    # submission = submission_df.append(submission_df_image)
    # submission.to_csv(PREDICTIONS_DIR + 'submission_onlyHighest_' + modelFile.split('_')[0] + '_' + modelFile.split('_')[-1].replace('.h5', '') + '.csv', index=False)
    # print('FINISHED ' + modelFile)
