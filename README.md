# Display and Product Detection

## Description
This is my capstone project for a data science bootcamp (The Data Incubator). It detects product facings, design features of displays and shelves in images of displays, counts them and saves the counts with the store ID, where the image was taken, to an excel file. More specifically, it turns an image of a display like

![This is an image showing a display.](readme_images/pretty_display.jpg)

into

![This is an image showing a spreadsheet.](readme_images/example_excel.jpg)

The project follows the TensorFlow Object Detection API Tutorial. The GitHub repository does not include images and annotation. To train the model on a different set of images, load them into TensorFlow/workspace/training_products_shelf/images and annotate as explained in the TensorFlow tutorial.

In addition, the project adds information about the location of the displays using the Google Maps API.

## Purpose of the Project

My project analyzes product displays of a particular promotion for a specific brand (not Pré de Provence). The aim is to answer the following questions:

* Where the correct displays put up for the specific promotion?
* Where the old displays replaced by the new ones?
* Where the displays correctly stacked?

## Results

This section is to demonstrate what sort of insights we can generate from the spreadsheet. Note that all the results are specific to my data set.

### Type of Display
Different promotions use different types of displays. The displays differ with respect to seasonal design features. I call the design feature that is characteristic for this promotion Design 1. The following bar chart shows how many displays with the different design features were put up by different retailers during this promotion.

![This is an image showing the shares of display designs by retailer.](readme_images/correct_display.jpg)

Insight 1: Retailer 3 - that used a different display assembly service - put up more of the correct displays compared to the other retailers.

### Old vs. New Displays
The old displays were supposed to be replaced by displays with a new design. The following bar chart shows how many displays with the new design compared to displays with the old design were put up by the different retailers during this promotion.

![This is an image showing the shares of old and new displays by retailer.](readme_images/new_vs_old.jpg)

Insight 2: Retailer 3 - that used a different display assembly service - put up more new displays than the other retailers.

### Product Facings
This brand recommended placing two specific products on the displays, which I call Product 1 and Product 2. The number of facings varied widely across displays. However, the majority of displays followed a similar layout for each retailer. The following bar chart compares the number of facings of Product 1 and Product 2 on a typical shelf for different retailers.

![This is an image showing the shares of two products on a typical shelf by retailer.](readme_images/typical_shelf.jpg)

Insight 3: Retailer 3 - that happens to have the most professional data science team - displays only one of the two recommended products.


### Other Products
All retailers included other products on some of the displays. The following point map shows the percentage of displays that include other products in different provinces of South Africa for two retailers. The size of the bubbles correspond to the number of images taken in the respective province.

![This is an image showing the shares of two products on a typical shelf by retailer.](readme_images/correct_products.jpg)

Insight 4: Retailer 3 and Retailer 4 included other products on almost all displays in the eastern provinces. This might reflect differences in tastes between customers in the east and customers in the west.

Insight 5: Gauteng is the largest market. The fieldforce obviously failed to collect images of all displays in Gauteng.

## Annotation

![This is an image showing annotation.](readme_images/example_annotation.jpg)

I annotate the data myself with LabelImg making sure that...

* ... the bounding boxes contain the entire object and are tight.
* ... occluded objects are labelled including the occluded part.
* ... all objects in the images are labelled.

## Relabelling of Annotation
I use this repo for different purposes. I have a script that relabels the train and test data (relabel_train.ipynb) as well as a script that relabels the data for analysis (relabel_analysis.ipynb). The relation between original and new labels are saved in npy-files.

## Annotation Check
The script annotation_check.ipynb checks whether all images have been labelled and plots a random sample of images with annotation.

## Object Detection
The script main.ipynb follows the TensorFlow Object Detection API Tutorial and trains the model. The script tensorflow_to_excel.ipynb loads the model and detects products, design features of displays and shelves. The detected objects are saved in an excel-file.

## Location
The script geo_info.ipynb looks up store location information using an excel table with additional information about the images. I add the address, longitude and latitude of the stores to the excel-file with the detected products and design features using the Google Maps API. I take the ZIP code from the address to determine the province.

## Display Analysis
The script display_analysis.ipynb explores my data set.
