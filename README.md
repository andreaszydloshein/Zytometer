# Zytometer

The Zytometer script is a Python tool designed for the automated analysis of fluorescence microscopy images, particularly focusing on the quantification of neutrophil populations. This script aids in the identification, counting, and analysis of neutrophils within fluorescence images, providing valuable data for various biological and medical research applications.

## Compatibility

The script is compatible with Python 3.x environments and relies on several external libraries, including OpenCV, NumPy, Matplotlib, Pandas, and scikit-image. Ensure that these libraries are installed in your Python environment before running the script.

## Accepted Images

The Zytometer script accepts fluorescence microscopy images in the following formats:

- **TIF**: Tagged Image File Format (TIF) files, commonly used in microscopy imaging due to their lossless compression and support for metadata.
- **JPEG**: Joint Photographic Experts Group (JPEG/JPG) files, providing a widely compatible format for images with moderate compression.

Ensure that your images are in either TIF or JPEG format before running the script.

## Output Data

After processing the input images, the Zytometer script generates comprehensive output data, including:

- **Neutrophil Counts**: Total counts of neutrophils identified in each region of interest.
- **Change in Integrated Fluorescence**: Changes in integrated fluorescence intensity within specified regions.
- **Total Integrated Fluorescence per Circle**: Quantification of fluorescence intensity within circular regions of interest.

The output data is stored in CSV (Comma-Separated Values) format, facilitating easy analysis and visualization using various data analysis tools.

## Workflow and Settings

The typical workflow for using the Zytometer script involves the following steps:

1. **Select Folder**: Choose the folder containing the fluorescence microscopy images you want to analyze.
2. **Specify Image Extensions**: Provide the file extensions for both Brightfield (BF) and enhanced Green Fluorescent Protein (eGFP) images.
3. **User Input**: Manually select the regions of interest within the images using a graphical interface.
4. **Image Processing**: The script automatically processes the selected images, identifying neutrophils and quantifying fluorescence intensity.
5. **Data Export**: Export the analyzed data to CSV files for further analysis and interpretation.

Additionally, you can adjust various settings within the script to fine-tune the analysis process according to your specific requirements.

## Sholl Analysis and Distance Measurements

The Zytometer script incorporates Sholl analysis, a method used to quantify neuronal dendritic arborization, adapted here for fluorescence image analysis. Additionally, the script calculates distance measurements between specified points of interest, aiding in the characterization of spatial distributions and relationships within the image.

## Exported Data

The Zytometer script exports the analyzed data into multiple CSV files, including:

- **Total_data.csv**: Contains comprehensive data on neutrophil counts, integrated fluorescence intensity, and changes in fluorescence intensity across different regions of interest.
- **Neutrophils_per_Bin.csv**: Provides detailed information on neutrophil counts per region of interest.
- **Group_Compilation.csv**: Compiles data on changes in integrated fluorescence intensity for group-level analysis.

These CSV files enable further exploration and visualization of the analyzed data using external data analysis tools and software.
