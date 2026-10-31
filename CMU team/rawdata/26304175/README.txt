----------------------------------------
GENERAL INFORMATION
----------------------------------------

1. Title of Dataset:  Dataset of Porosity of Powder Bed Fusion - Laser Beam Ti-6Al-4V from X-ray Micro Computed Tomography

2. Author Information

Author Contact Information
    Name: Justin P. Miner
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: jpminer@andrew.cmu.edu


Author Contact Information
    Name: Sneha Prabha Narra
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: snarra@andrew.cmu.edu

---------------------------------------
DATA & FILE OVERVIEW
---------------------------------------

Directory of Files:

   A. Filename:  poredf.csv
   
      Short description:  This CSV file contains all of the pores in the form of size, aspect ratio, and sphericity from X-ray Micro Computed Tomography scans. Note this file should not be saved in excel because it contains too many rows. Doing so would result in data loss.


   B. Filename:  parameters.csv   
   
      Short description:  This CSV file contains the fit EVS parameters for each of the samples referenced in poredf.csv

File Naming Convention: data.csv  


----------------------------------------------------------------------------------------------------------
DATA DESCRIPTION FOR: poredf.csv
----------------------------------------------------------------------------------------------------------

1. Number of variables: 4


2. Number of cases/rows: 1024745


3. Missing data codes: The dataset has no missing data, but in the case of missing codes, the dataset would use "-1" to denote missing data. 

4. Variable List
	
	A. Name: Name
	
       Description: Denotes the sample that porosity came from by a unique identifier

	B. Name: Equivalent Spherical Diameter (um)
	
       Description: Denotes the diameter of a sphere with equivalent volume  
					
	C. Name: Aspect Ratio
	
       Description: Denotes the ratio of smallest to longest dimensions
					
	 D. Name: Sphericity
	 
       Description: Denotes the spherity of a given pore


--------------------------------------------------------------------------------------------------------------
DATA DESCRIPTION FOR: parameters.csv 
--------------------------------------------------------------------------------------------------------------

1. Number of variables: 18


2. Number of cases/rows: 39


3. Missing data codes: This dataset has missing data codes for the Powder because it does not have sample related parameters. These parameters are indicated by a blank entry. Depending on the software, this may be interpreted as blank, NULL, or 0.


4. Variable List

    A. Name: Name
	
       Description: Denotes the sample that porosity came from by a unique identifier. Links to poredf.csv

    B. Name: Power (W)
	
       Description: Denotes the laser power in watts.
					
    C. Name: Velocity (mm/s)
	
       Description: Denotes the laser scanning velocity in mm/s
					
    D. Name: Spot Size (um)
	
       Description: Denotes the laser spot size in micrometers
    
    E. Name: Hatch Spacing (um)
	
       Description: Denotes the hatch spacing or distance between adjacent scan lines in micrometers

    F. Name: Layer Height (um)
	
       Description: Denotes the amount of powder added per layer in micrometers

    G. Name: Geometry
	
       Description: Denotes the geometry. AX: Axial, 4PB: 4-point bend

    H. Name: Volume (mm^3)
	
       Description: Denotes the volume scanned using X-ray Micro Computed Tomography in millimeters cubed

    I. Name: Threshold (um)
	
       Description: Denotes the chosen threshold from the extreme value statistics fit in micrometers

    J. Name: Scale
	
       Description: Denotes the fit scale parameter

    K. Name: STDScale
	
       Description: Denotes the standard deviation in the fit scale parameter

    L. Name: Shape
	
       Description: Denotes the fit shape parameter

    M. Name: STDShape
	
       Description: Denotes the standard deviation of the fit scale parameter

    N. Name: CovShapeScale
	
       Description: Denotes the covariance between the shape and scale parameter

    O. Name: Rate (1/mm^3)
	
       Description: Denotes the rate of upper tail pore occurrence in pores per millimeter cubed.

    P. Name: STDRate
	
       Description: Denotes the standard deviation of the rate estimator.

    Q. Name: X (mm)
	
       Description: Denotes the build plate location from the recoater in mm. Uses EOSM290 coordinate system.

    R. Name: Y (mm)
	
       Description: Denotes the build plate location from the gas flow in mm. Uses EOSM290 coordinate system.
					


-------------------------------------------------------
METHODOLOGICAL INFORMATION
-------------------------------------------------------

1. Software-specific information:

Name: Python
Version: 3.9.13 
System Requirements: Windows, macOS, Linux
Open Source? (Y/N):  Y

Additional Notes: Data was processed and extracted using pandas and numpy.

Name: Dragonfly Pro Academic Version
Version: 2022.2.0.12227
System Requirements: Windows, macOS, Linux
Open Source? (Y/N):  N

Additional Notes: The X-ray Micro Computed Tomography Scans were segmented and processed using this software.


2. Equipment-specific information:

Manufacturer: Lambda
Model: Custom Build: 128 GB RAM, 2 RTX4090 GPU, AMD Threadripper CPU

Additional Notes: The data was collected on this desktop.


Manufacturer: Razer
Model: Blade Laptop Base 2018

Additional Notes: The data was cleaned and formatted on this laptop. 


3. Date of data collection: 2024/4/1 - 2023/12/31

--------------------------------------------------
NOTES ON REPRODUCIBILITY 
--------------------------------------------------

The raw computed tomography scans and segmented scans are available upon request. These scans are unfeasible to host due to the large amount of data contained within.
