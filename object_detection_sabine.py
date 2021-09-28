# Object Detection

# This is a collection of functions that I use in connection with object detection.

# Imports
import os
import random
import collections
import pandas as pd
import cv2 as cv
import tensorflow as tf
from matplotlib import patches
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

####################################################################################################
# XML to Pandas DataFrame for image and annotation inspection

def col_dict(etree, x_path):
    """This function creates a dictionary with object id and (sub-)child in x_path.

    Args:
        etree: element tree
        x_path: xml xpath
        
    Return:
        col_dict: dictionary with object id as key and (sub-)child text as value
    """
    obj_counter = 1
    col_dict = {}
    for el in etree.findall(x_path):
        col_dict['object'+str(obj_counter)] = el.text
        obj_counter += 1
    return col_dict

def xml_to_df(file_path):
    """This function converts the xml-file with path path_name into a Pandas DataFrame."""
    etree = ET.parse(file_path)

    pd_dict = {}
    col_name_xpath_dict = {'name': 'object/name',
                           'xmin': 'object/bndbox/xmin',
                           'xmax': 'object/bndbox/xmax',
                           'ymin': 'object/bndbox/ymin',
                           'ymax': 'object/bndbox/ymax'}

    for key, x_path in col_name_xpath_dict.items():
        pd_dict[key] = col_dict(etree, x_path)

    df = pd.DataFrame(pd_dict)

    pd_dict = {'filename': 'filename',
               'width': 'size/width',
               'height': 'size/height',
               'depth': 'size/depth'}

    for key, x_path in pd_dict.items():
        df[key] = etree.find(x_path).text

    return df

def all_xml_to_df(folder_path):
    """This function reads the filename, size, and object names with the bounding boxes from the XML-files in the train folder
       and stores them in a Pandas dataframe.

    Arg:
        path: string with path to train folder
    Return:
        df: Pandas dataframe
    """
    df = None
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            if df is None:
                df = xml_to_df(folder_path+'/'+filename)
            else:
                df = df.append(xml_to_df(folder_path+'/'+filename))

    return df

####################################################################################################
# Pandas DataFrame: Overview of annotation information

def annotation_overview(df):
    """This function provides some insights into the content of the xml-files with the annotations.
    It should be used in conjunction with the function all_xml_to_df.
    
    Arg:
        df: Pandas dataframe
        
    Return:
        None: prints number of images, size of the images, and number of labels
    """
    
    if 'filename' in df.columns:
        nr_img = len(df['filename'].unique())
        print(f'There are {nr_img} images.')
        print()

    if 'width' in df.columns:
        width_ls = df['width'].unique()
        width_height_ls = [(w, h) for w in width_ls for h in df.loc[df['width']==w]['height'].unique()]
        print('The images are of the following sizes:')
        for w, h in set(width_height_ls):
            print(w, 'x', h)
        print()
    
    if 'name' in df.columns:
        print('The labels are:')
        for label_name in df['name'].unique():
            print(label_name)

    return

####################################################################################################
# Annotation check: each jpg with an xml of same name

def annotation_check(path_name):
    """This function checks if the number of jpg-files in the folder under path_name equals the number of xml-files.

    Arg:
        path_name: string with path to folder with jpg- and xml-files
        
    Return:
        None: prints result
    """

    jpg_ls = []
    xml_ls = []

    for filename in os.listdir(path_name):
        if filename.endswith('.jpg'):
            jpg_ls.append(filename[:-4])
        elif filename.endswith('.xml'):
            xml_ls.append(filename[:-4])

    jpg_count = collections.Counter(jpg_ls)
    xml_count = collections.Counter(xml_ls)


    if len(jpg_count) == 0 & len(xml_count) == 0:
        print('There are no jpg- and xml-files in the folder.')
        return

    if len(jpg_count) == 0:
        print('There are no jpg-files in the folder.')
        return

    if len(xml_count) == 0:
        print('There are no xml_files in the folder.')
        return

    if (jpg_count == xml_count):
        print('All jpg-files have an xml-file with the same name.')
        return

    if len((jpg_count - xml_count).keys()) > 0:
        print('The following jpg-files have no xml-file with the same name: ')
        for key in (jpg_count - xml_count).keys():
            print(str(key)+'.jpg')

    if len((xml_count - jpg_count).keys()) > 0:
        print('The following xml-files have no jpg-file with the same name: ')
        for key in (xml_count - jpg_count).keys():
            print(str(key)+'.xml')

    return

####################################################################################################
# Plot some sample images

def plot_sample(path_name, nr_images, max_width, max_height):
    """This function plots a number of images that are randomly chosen from the folder under path_name.
    
    Args:
        path_name: string with path to folder with jpg-files
        nr_images: int for number of images in plot
        max_width: int for maximal width of images in pixels
        max_height: int for maximal height of images in pixels
        
    Return:
        None: prints images
    """

    img_ls = [filename for filename in os.listdir(path_name) if filename.endswith('.jpg')]
    img_ls = random.sample(img_ls, nr_images)

    columns = 4
    rows = int(nr_images / columns) + (nr_images % columns > 0)
    fig = plt.figure(figsize=(columns*2+columns/4-1, rows*2+rows/4-1), dpi=max(max_width, max_height)/2)

    for img_nr in range(nr_images):
        img_name = img_ls[img_nr]
        image = cv.imread(path_name+'/'+img_name)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, img_nr+1)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(img_name, fontsize=6)

    return

####################################################################################################
# Plot some sample images with bounding boxes

def plot_sample_bndbox(path_name, nr_images, bnd_box_df):
    """This function plots a number of images that are randomly chosen from the folder under path_name
    including bounding boxes. It should be used in conjunction with the function all_xml_to_df.
    
    Args:
        path_name: string with path to folder with jpg-files
        nr_images: int for number of images in plot
        bnd_box_df: Pandas dataframe with filename, size, object names and bounding boxes from XML-files
        
    Return:
        None: prints images
    """

    img_ls = [filename for filename in os.listdir(path_name) if filename.endswith('.jpg')]
    img_ls = random.sample(img_ls, nr_images)

    columns = 4
    rows = int(nr_images / columns) + (nr_images % columns > 0)
    max_width = max(pd.to_numeric(bnd_box_df['width']))
    max_height = min(pd.to_numeric(bnd_box_df['height']))
    fig = plt.figure(figsize=(columns*2+columns/4-1, rows*2+rows/4-1), dpi=max(max_width, max_height)/2)

    for img_nr in range(nr_images):
        img_name = img_ls[img_nr]
        img_df = bnd_box_df[bnd_box_df['filename'] == img_name]
        image = cv.imread(path_name+'/'+img_name)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows, columns, img_nr+1)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(img_name, fontsize=6)
        for idx in img_df.index:
            obj_width = float(img_df.loc[idx,'xmax']) - float(img_df.loc[idx,'xmin'])
            obj_height = float(img_df.loc[idx,'ymax']) - float(img_df.loc[idx,'ymin'])
            obj_x = float(img_df.loc[idx,'xmin'])
            obj_y = float(img_df.loc[idx,'ymin'])
            ax.add_patch(patches.Rectangle((obj_x, obj_y), obj_width, obj_height, fill=False, edgecolor='lime', lw=0.5))

    return

####################################################################################################
# Jpg to Numpy array for TensorFlow graph

def jpg_to_np(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

####################################################################################################
# Detection of objects in an image

@tf.function
def detect_fn(image, model_for_detection):
    """Detect objects in image.

    Args:
        image: TensorFlow tensor of an image
        model_for_detection: TensorFlow object detection model

    Return:
        detections: dictionary with detection information    
    """

    image, shapes = model_for_detection.preprocess(image)
    prediction_dict = model_for_detection.predict(image, shapes)
    detections = model_for_detection.postprocess(prediction_dict, shapes)

    return detections

####################################################################################################
# Save np array with image and all detected objects

def save_image_np_with_all_detections(path, model_for_detection, category_index):
    """This function converts the image under path into a numpy array, detects objects in the image with the model_for_detection,
    and saves the numpy array with the image and detected objects.
    
    Args:
        path: path to jpg
        model_for_detection: TensorFlow model loaded with model_builder
        category_index: dictionary mapping categories to labels

    Return:
        None: saves numpy array with the image and detected objects
    """

    img_np = jpg_to_np(path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor, model_for_detection)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = img_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                                                image_np_with_detections,
                                                detections['detection_boxes'],
                                                detections['detection_classes']+label_id_offset,
                                                detections['detection_scores'],
                                                category_index,
                                                use_normalized_coordinates=True,
                                                max_boxes_to_draw=200,
                                                min_score_thresh=.30,
                                                agnostic_mode=False,
                                                skip_labels=True,
                                                skip_scores=True)
    np.save('image_np_with_detections.npy', image_np_with_detections)

    return

####################################################################################################
# Save np array with image and one detected objects

def save_image_np_with_one_detection(path, model_for_detection, category_index, obj_nr):
    """This function converts the image under path into a numpy array, detects one object, which is selected by obj_nr,
    in the image with the model_for_detection, and saves the numpy array with the image and the detected object.
    
    Args:
        path: path to jpg
        model_for_detection: TensorFlow model loaded with model_builder
        category_index: dictionary mapping categories to labels

    Return:
        None: saves numpy array with the image and the detected object
    """

    img_np = jpg_to_np(path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor, model_for_detection)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    one_detection = {}
    one_detection['detection_boxes'] = np.array([detections['detection_boxes'][obj_nr]])
    one_detection['detection_classes'] = np.array([detections['detection_classes'][obj_nr]])
    one_detection['detection_scores'] = np.array([detections['detection_scores'][obj_nr]])
    one_detection['num_detections'] = 1
    label_id_offset = 1
    image_np_with_detections = img_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                                                image_np_with_detections,
                                                one_detection['detection_boxes'],
                                                one_detection['detection_classes']+label_id_offset,
                                                one_detection['detection_scores'],
                                                category_index,
                                                use_normalized_coordinates=True,
                                                max_boxes_to_draw=200,
                                                min_score_thresh=.30,
                                                agnostic_mode=False,
                                                skip_labels=True,
                                                skip_scores=True)
    np.save('image_np_with_detections.npy', image_np_with_detections)

    return

####################################################################################################
# Object detection from jpg-file

def detection_from_jpg(jpg_file_path, detection_model):
    """This function detects objects with a TensorFlow-model.
    Args:
        jpg_file_path: path to jpg
        model_for_detection: TensorFlow model loaded with model_builder

    Return:
        all_detections: dictionary with information about detected objects
    """

    img_np = jpg_to_np(jpg_file_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    all_detections = detect_fn(input_tensor, detection_model)
    num_detections = int(all_detections.pop('num_detections'))
    all_detections = {key: value[0, :num_detections].numpy() for key, value in all_detections.items()}
    all_detections['num_detections'] = num_detections
    all_detections['detection_classes'] = all_detections['detection_classes'].astype(np.int64)

    return all_detections

####################################################################################################
# Select detections above a certain threshold score

def detection_selection(tf_detections, threshold):
    """This function selects the objects that were detected with TensorFlow that have a detection score above the threshold.
    
    Args:
        tf_detections: dictionary with information about detected objects
        threshold: float
        
    Return:
        detections: dictionary with information about detected objects that have detection score above threshold"""

    detections = {}
    detections['detection_boxes'] = tf_detections['detection_boxes'][tf_detections['detection_scores'] > threshold]
    detections['detection_scores'] = tf_detections['detection_scores'][tf_detections['detection_scores'] > threshold]
    detections['detection_classes'] = tf_detections['detection_classes'][tf_detections['detection_scores'] > threshold]
    detections['raw_detection_boxes'] = tf_detections['raw_detection_boxes'][tf_detections['detection_scores'] > threshold]
    detections['raw_detection_scores'] = tf_detections['raw_detection_scores'][tf_detections['detection_scores'] > threshold]
    detections['detection_multiclass_scores'] = tf_detections['detection_multiclass_scores'][tf_detections['detection_scores'] > threshold]
    detections['detection_anchor_indices'] = tf_detections['detection_anchor_indices'][tf_detections['detection_scores'] > threshold]
    detections['num_detections'] = sum(tf_detections['detection_scores'] > threshold).item()

    return detections