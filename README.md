# CyTOF-predictor
### XGBoost based CyTOF cell population predictor
For execution of prediction scripts, model weights are necessary. Download model weights from: ... and put it into models directory into directory with prediction.py script. 

Expected files locations
```
classifier/
├── models/
│   └── <model_weights>
└── prediction.py
```
Model weights specific for MoMyB CyTOF panel can be downloaded from: 
http://www.embnet.sk/project/cytof/

Isotopes used in panel:
#### Windows install and default execution:
```
cd C:\location_of_classifier_directory
py -m venv venv\
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas xgboost scikit-learn openpyxl
python -c "import numpy, pandas, xgboost, sklearn, openpyxl; print('OK')"
python predict.py --model_type full --input data.csv --input_format csv --output predictions.csv
deactivate
```

#### macOS install and default execution:
```
cd ~/location_of_classifier_directory
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas xgboost scikit-learn openpyxl
python -c "import numpy, pandas, xgboost, sklearn, openpyxl; print('OK')"
python predict.py --model_type full --input input.csv --input_format csv --output predictions.csv
deactivate
```

#### All available parameters:
```
python predict.py \
  --model_type {flat,lineage,full}
  --input input.csv
  --input_format {csv,excel}
  --output output.csv
  --chunk_size 100000 # default 100,000, process cells in chunks for faster prediction
  --predict_chunk_size 100000 # default 100,000, process cells in chunks for faster prediction
  --nthread #number of threads of cpu for prediction
  --predictor {cpu_predictor,gpu_predictor} #gpu acceleration
  --fillna_value FILLNA_VALUE #replace nans with specific value.
  --keep_input_columns  # keep all original input columns in output, results in large file and slower output.
  --keep_first_n_columns n_columns, keep selected columns as a metadata.
```
## License

MIT License. See the LICENSE file for details.
