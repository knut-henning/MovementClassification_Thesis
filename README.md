# MovementClassification_Thesis

Code and data for accelerometer data preprocessing and machine learning to classify movement in cows and calfs. Special thanks to Nofence!

### Caution
To be able to run the python scrips a folder with accelerometerdata is needed with the name "akselerometer_kalvelykke", provided by Nofence with permission.

### Files and Folders:
- **MovementClassification_Thesis**
  - **atferd_kalvelykke**                                         # Folder for classification data
    - **Atferds_registreringer_juni-aug_2020_SAMLET.csv**         # Classifications provided by Nofence. (multiclass)
    - **Atferds_registreringer_juni-aug_2020_SAMLET_RYDDET.csv**  # Classification provided by Nofence, rough cleaned for input to python script. (multiclass)
    - **Egenklassifisert_atf_bevegelse.csv**                      # Classification done by Knut-Henning Kofoed for different movement categories. (multiclass)
    - **Egenklassifisert_atf_drøvtygging.csv**                    # Classification done by Knut-Henning Kofoed for "drøvtygging". (binary classification)
  - **Datarydding kalvelykke.ipynb**                              # Notebook, Data preprocessing. (old code for testing, further development in .py script)
  - **Maskinlæring kalvelykke kalv.ipynb**                        # Notebook, Machinelearning on calf data. (old code for testing, further development in .py script)
  - **Maskinlæring kalvelykke ku.ipynb**                          # Notebook, Machinelearning on cow data. (old code for testing, no further development currently planned)
  - **Posisjonsdata kalvelykke.ipynb**                            # Notebook, Visualizing positional data. (old code for testing, no further development currently planned)
  - **drøvtygging_kalvelykke.py**                                 # Machinelearning on classifying binary "drøvtygging". (Still under development and testing)
  - **egenklassifisert_kalvelykke.py**                            # Machinelearning on own classified multiclass movement. (Still under development and testing)
  - **machinelearn_kalvelykke.py**                                # Collection of functions to be used for machine learning.
  - **prepro_kalvelykke.py**                                      # Collection of functions to be used for preprocessing.
  - **visualize_kalvelykke.py**                                   # Collection of functions to be used for visualizing data and results.
