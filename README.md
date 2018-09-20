From what I can tell based on the commit history and the paper, the 'Gesture Recognition/mainMLScript.py' carries out the 
Gesture Recognition phase on some sample data. 

The sample data is already windowed to 128 data points per window, and labeled with a patient ID 
and classification identifier. See 'reading-csv-file.txt' for column details.

The processing includes all steps included in 'Gesure Recognition/featureExtraction.py' file. The 
primary function which extracts the features is named "main" in 'Gesture Recognition/featureExtraction.py'.

'raw to sequences.py' script corresponds to the offline model building framework pictured in 'Figure 2' of the paper. The script
carries out the window extraction step (3.B in the paper). It then carries out feature extraction 
described in section 3.C in the paper. It then performs the K-Means data clustering intended for activity-phase recognition 
described in section 3.D of the paper. Then it carries out the Random Forest activity recognition described in section 3.E.



Dependency Setup
----------------

These scripts have numerous dependencies. I'll list commands you can use on Ubuntu to install them below:

* sudo apt install python-tk
* pip install pandas matplotlib scipy scikit-learn numpy

In 'raw to sequences.py' I commented out the stuff involving weka imports because it doesn't appear to actually be used in the script.