# CNN-LSTM Model for Surgical Phase Detection in Prostatectomy

A CNN-LSTM model is trained for surgical phase detection in prostatectomy. 


## Metrics

* Overall accuracy: $78.4$%
* Overall precision: $67.3$%
* Accuracy and recall for surgical phases:

| Phase | Precision (%) | Recall (%) |
| ----- | ------------- | ---------- |
| BN    | 81.1          | 74.5       |
| BNU   | 95.4          | 98.5       |
| DVC   | 65.3          | 30.5       |
| LN    | 86.9          | 67.5       |
| NVB   | 68.0          | 80.7       |
| PA    | 66.3          | 52.7       |
| SVVD  | 75.6          | 64.6       |


## Train

   ```shell
   python convert_data_to_vdlp.py path/to/videos frames/save/path 1
   ```

   ```shell
   python main.py --resume_from saved_models/baseline.pth saved_models/v2.pt  --only_test
   ```
   
   
 ## License
 
 This project is licensed under the MIT License
