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
Nd145Di, Dy164Di, Er167Di, Gd155Di, Yb174Di, Er168Di Cd116Di, Dy163Di, Lu175Di, Yb173Di, Nd148Di, Nd150Di, Sm149Di, Cd112Di, Cd111Di, Gd160Di, Nd142Di, Cd114Di, Yb171Di, Gd156Di, Yb176Di, Cd113Di, Dy162Di, Sm147Di, Dy161Di, Er166Di, Nd146Di, Ho165Di, Gd158Di, Nd144Di, Tb159Di, Y89Di, Tm169Di, Nd143Di, Sm152Di, Pr141Di, Eu151Di, Er170Di, Eu153Di, Yb172Di, Sm154Di
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
