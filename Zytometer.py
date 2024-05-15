

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from IPython.display import display, Image
import csv 
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage import measure
import os
import pandas as pd
import keyboard
from tqdm import tqdm


def draw_circles(image, props):
      
    """
    Draw circles around the centroids of detected objects in the image.

    Args:
        image (numpy.ndarray): Input image.
        props (list): List of region properties obtained from skimage.measure.regionprops.

    Returns:
        numpy.ndarray: Image with circles drawn around the centroids.
    """
    
    image_with_circles = image.copy()
    for p in props:
        center = (int(p.centroid[1]), int(p.centroid[0]))
        radius = int(np.sqrt(p.area / np.pi))
        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)
    return image_with_circles



def draw_contours(image, props):
    
    """
    Draw contours around the detected objects in the image.

    Args:
        image (numpy.ndarray): Input image.
        props (list): List of region properties obtained from skimage.measure.regionprops.

    Returns:
        numpy.ndarray: Image with contours drawn around the detected objects.
    """
    
    image_with_contours = image.copy()
    for p in props:
        center = (int(p.centroid[1]), int(p.centroid[0]))
        radius = int(np.sqrt(p.area / np.pi))
        cv2.circle(image_with_contours, center, radius, (0, 255, 0), 2)
    return image_with_contours



def draw_total_fluorescence_circles(image, center, radii, total_fluorescence_per_circle):
   
    """
    Draw circles representing total fluorescence at different radii.

    Args:
        image (numpy.ndarray): Input image.
        center (tuple): Center coordinates of the circles.
        radii (list): List of radii for the circles.
        total_fluorescence_per_circle (list): Total fluorescence per circle.

    Returns:
        numpy.ndarray: Image with circles representing total fluorescence drawn.
    """   
    
    image_with_circles = image.copy()
    
    for i, (range_start, range_end) in enumerate(radii):
        radius = (range_start + range_end) // 2
        
        total_fluorescence = total_fluorescence_per_circle[i]
        cv2.circle(image_with_circles, center, radius, (255, 0, 0), 2)  
        cv2.putText(image_with_circles, f'{total_fluorescence:.2f}', (center[0] + radius + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return image_with_circles



def count_dots_in_circles(image, center, radii, props, prev_counts):
        
    """
    Count dots within circular regions of interest in the image.

    Args:
        image (numpy.ndarray): Input image.
        center (tuple): Center coordinates of the circles.
        radii (list): List of radii for the circles.
        props (list): List of region properties obtained from skimage.measure.regionprops.
        prev_counts (list): Previous counts of dots in the circles.

    Returns:
        tuple: Counts of dots, new counts, total fluorescence per circle, and image with circles.
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    dot_counts = []
    new_counts = []
    total_fluorescence_per_circle = []  
    image_with_circles = image.copy()

    for i, (range_start, range_end) in enumerate(radii):
        radius = (range_start + range_end) // 2

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        masked_image = cv2.bitwise_and(thresholded, thresholded, mask=mask)

        labeled_image, num_labels = label(masked_image, connectivity=2, return_num=True)

        neutrophil_count = 0
        new_neutrophil_count = 0
        total_fluorescence = 0

        for p in props:
            for j in range(1, num_labels + 1):
                if j in labeled_image[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]:
                    neutrophil_count += 1
                    total_fluorescence += np.sum(p.intensity_image)  
                    
        dot_counts.append(neutrophil_count)

        if prev_counts[i] is None:
            new_neutrophil_count = neutrophil_count
        else:
            new_neutrophil_count = neutrophil_count - prev_counts[i]
            if new_neutrophil_count < 0:
                new_neutrophil_count = 0



        new_counts.append(new_neutrophil_count)
        total_fluorescence_per_circle.append(total_fluorescence)

        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)

    return dot_counts, new_counts, total_fluorescence_per_circle, image_with_circles



def get_user_input(image, image_path):
     
    """
    Get user input by allowing them to select points on an image.

    Args:
        image (numpy.ndarray): Input image.
        image_path (str): Path to the image file.

    Returns:
        tuple: Tuple containing the coordinates of the selected points.
    """
           
    clicked = False
    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked, points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            if len(points) == 2:
                clicked = True
                cv2.destroyAllWindows()

    cv2.namedWindow('Select Points')
    cv2.setWindowTitle('Select Points', os.path.basename(image_path))
    cv2.setMouseCallback('Select Points', mouse_callback)
    cv2.imshow('Select Points', image)
    cv2.waitKey(0)

    if len(points) != 2:
        print("Two points were not selected. Exiting")
        return None, None

    return points[0], points[1]



def process_image(image, image_path, dir_path, start_point, end_point):
        
    """
    Process an image to analyze neutrophil activity and fluorescence.

    Args:
        image (numpy.ndarray): Input image.
        image_path (str): Path to the image file.
        dir_path (str): Directory path.
        start_point (tuple): Starting point coordinates.
        end_point (tuple): Ending point coordinates.
    """
        
    if image is None:
        print(f"Invalid image file: {image_path}")
        return

    distance = int(np.linalg.norm(np.array(start_point) - np.array(end_point)))
    num_radii = distance // 50
    radii = [(i * 50, (i + 1) * 50) for i in range(num_radii)]

    image_with_sholl = image.copy()
    cv2.line(image_with_sholl, start_point, end_point, (0, 255, 0), 2)
    for radius in radii:
        cv2.circle(image_with_sholl, start_point, radius[1], (0, 255, 0), 2)
        
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image_with_sholl, cv2.COLOR_BGR2RGB))
    # plt.title("eGFP Image with Sholl Circles")
    # plt.axis('off')
    # plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    labelarray, particle_count = ndimage.label(gray)

    maxValue = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY
    blockSize = 5
    C = -3
    im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C)

    labelarray, particle_count = ndimage.label(im_thresholded)

    props = measure.regionprops(labelarray, intensity_image=gray)

    total_fluorescence = 0

    total_fluorescence_per_circle = []
    for (range_start, range_end) in radii:
        radius = (range_start + range_end) // 2

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, start_point, radius, 255, -1)

        masked_image = cv2.bitwise_and(gray, gray, mask=mask)

        total_fluorescence_per_circle.append(np.sum(masked_image))

    image_with_fluorescence = image.copy()

    for p in regionprops(labelarray, intensity_image=gray):
        total_fluorescence += np.sum(p.intensity_image)
        image_with_fluorescence[p.coords[:, 0], p.coords[:, 1]] = [255, 0, 0]

    for (range_start, range_end) in radii:
        radius = (range_start + range_end) // 2

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, start_point, radius, 255, -1)

        masked_image = cv2.bitwise_and(gray, gray, mask=mask)

        total_fluorescence_per_circle.append(np.sum(masked_image))

    image_with_circles = draw_circles(image, props)
    dot_counts, new_counts, total_fluorescence_per_circle, image_with_circles = count_dots_in_circles(image_with_circles, start_point, radii, props, [0] * len(radii))

    image_with_contours = draw_contours(image, props)

    bin_labels = [f"{range_start}-{range_end}" for (range_start, range_end) in radii]

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(dir_path, f'{filename}.Total_data.csv')

    with open(output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Bin', 'Neutrophil Numbers', 'Change in Integrated Fluorescence', 'Total Integrated Fluorescence per Circle'])

        bin_counts = [f"{range_start}-{range_end}" for (range_start, range_end) in radii]
        prev_counts = [0] * len(radii)
        prev_total_fluorescence = [0] * len(radii)
        for i, (bin_label, dot_count, total_fluorescence, circle_fluorescence) in enumerate(zip(bin_labels, dot_counts, total_fluorescence_per_circle, total_fluorescence_per_circle)):
            new_count = dot_count - prev_counts[i]

            if i > 0:
                change_in_fluorescence = total_fluorescence - prev_total_fluorescence[i - 1]
            else:
                change_in_fluorescence = total_fluorescence

            csv_writer.writerow([bin_label, dot_count, new_count, change_in_fluorescence, circle_fluorescence])
            prev_counts[i] = dot_count
            prev_total_fluorescence[i] = total_fluorescence

    df = pd.read_csv(output_path)
    df['Bin'] = bin_labels
    df['Neutrophils per Bin'] = df['Neutrophil Numbers'].diff().fillna(df['Neutrophil Numbers'])
    df.loc[df['Neutrophils per Bin'] < 0, 'Neutrophils per Bin'] = 0
    df.to_csv(output_path, index=False)

    print("Data has been written to:", output_path)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image_with_fluorescence, cv2.COLOR_BGR2RGB))
    # plt.title("eGFP Image with Labeled Fluorescence")
    # plt.axis('off')
    # plt.show()

    output_file_neutrophils = os.path.join(dir_path, f'{filename}.Neutrophils_per_Bin.csv')
    create_new_csv(dir_path, output_file_neutrophils, 'Neutrophils per Bin')

    output_file_fluorescence = os.path.join(dir_path, f'{filename}.Group_Compilation.csv')
    create_new_csv(dir_path, output_file_fluorescence, 'Group Compilation')

    avg_output_path = os.path.join(dir_path, 'average_data.csv')

    total_neutrophils = len(props)
    avg_neutrophil_size = sum(p.area for p in props) / len(props) if len(props) > 0 else 0
    avg_integrated_fluorescence = sum(p.mean_intensity for p in props) / len(props) if len(props) > 0 else 0
    total_integrated_fluorescence = total_fluorescence

    avg_data = [
        filename,
        total_neutrophils,
        avg_integrated_fluorescence,
        avg_neutrophil_size,
        total_integrated_fluorescence
    ]

    with open(avg_output_path, 'a', newline='') as avg_csv_file:
        csv_writer = csv.writer(avg_csv_file)
        csv_writer.writerow(avg_data)


        
def create_new_csv(input_folder, output_file, column_name):
       
    """
    Create a new CSV file from data in input CSV files.

    Args:
        input_folder (str): Input folder path.
        output_file (str): Output CSV file path.
        column_name (str): Name of the column to extract from input files.
    """
    
    csv_files = [file for file in os.listdir(input_folder) if file.lower().endswith('total_data.csv')]
    bins_column = None
    output_data = {}

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        if bins_column is None:
            bins_column = df['Bin']
            output_data['Bins'] = bins_column
        else:
            if len(df['Bin']) > len(bins_column):
                bins_column = df['Bin']
                output_data['Bins'] = bins_column

        output_data[file] = df.get(column_name, pd.Series(np.zeros(len(df))))

    output_df = pd.concat(output_data, axis=1)

    if 'Bins' in output_df.columns:
        output_df.drop(columns=['Bins'], inplace=True)

    output_df.insert(0, 'Bins', bins_column)

    output_df.to_csv(output_file, index=False)

    
    
def create_group_compilation_csv(input_folder, output_file):
        
    """
    Create a group compilation CSV file.

    Args:
        input_folder (str): Input folder path.
        output_file (str): Output CSV file path.
    """
     
    csv_files = [file for file in os.listdir(input_folder) if file.lower().endswith('total_data.csv')]
    bins_column = None
    output_data = {}

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        if bins_column is None:
            bins_column = df['Bin']
            output_data['Bins'] = bins_column
        else:
            if len(df['Bin']) > len(bins_column):
                bins_column = df['Bin']
                output_data['Bins'] = bins_column

        output_data[os.path.splitext(file)[0]] = df['Change in Integrated Fluorescence']

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)


def main():
       
    """
    Main function to execute the image processing pipeline.
    """
    
    Tk().withdraw()
    dir_path = askdirectory(title='Select the folder with images')
    if not dir_path:
        print("No folder selected. Exiting")
        return

    bf_file_extension = input("Enter the file extension for Brightfield images (e.g., 'ch02.tif'): ")
    eGFP_file_extension = input("Enter the file extension for eGFP images (e.g., 'ch00.tif'): ")

    bf_image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(bf_file_extension)]
    eGFP_image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(eGFP_file_extension)]

    if len(bf_image_files) == 0 or len(eGFP_image_files) == 0:
        print("No image files found in the selected folder. Exiting")
        return

    output_file_neutrophils = os.path.join(dir_path, 'average_data.csv')
    output_file_fluorescence = os.path.join(dir_path, 'Group_Compilation.csv')

    with open(output_file_neutrophils, 'w', newline='') as avg_csv_file:
        csv_writer = csv.writer(avg_csv_file)
        csv_writer.writerow([ 
            'File Name',
            'Total Neutrophils',
            'Average Integrated Fluorescence',
            'Average Neutrophil Size',
            'Total Integrated Fluorescence'
        ])

    avg_data_list = []

    for bf_image_file in tqdm(bf_image_files, desc='Processing BF Images', unit='image'):
        bf_image_path = os.path.join(dir_path, bf_image_file)

        start_point_bf, end_point_bf = get_user_input(cv2.imread(bf_image_path), bf_image_path)

        if start_point_bf is None or end_point_bf is None:
            continue

        print("Processing corresponding eGFP image...")
        eGFP_image_file = bf_image_file.replace(bf_file_extension, eGFP_file_extension)
        eGFP_image_path = os.path.join(dir_path, eGFP_image_file)

        start_point_eGFP, end_point_eGFP = start_point_bf, end_point_bf

        avg_data = process_image(cv2.imread(eGFP_image_path), eGFP_image_path, dir_path, start_point_eGFP, end_point_eGFP)

        if avg_data is not None:
            avg_data_list.append(avg_data)

    with open(output_file_neutrophils, 'a', newline='') as avg_csv_file:
        csv_writer = csv.writer(avg_csv_file)
        csv_writer.writerows(avg_data_list)

    print(f"Average data saved to {output_file_neutrophils}")

    create_group_compilation_csv(dir_path, output_file_fluorescence)
    print(f"Group compilation data saved to {output_file_fluorescence}")

    print("DONE!")

    
    
if __name__ == '__main__':
    main()
