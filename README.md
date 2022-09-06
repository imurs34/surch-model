### Architecture:

There are four main subnetworks that I am using

1. Pretrained RESNET50
2. LSTM with hidden dimension of 512 and sequence length of 10
3. Drop out layer followed by fully connected layer
4. Softmax layer

Two other important details are:

1. For training, I am using focal loss, because of class imabalance problem (one class, e.g, has more than 40% of the samples)
2. For testing, I am using an ensemble of two trained networks. Their probabilties for each class are averaged



### Training Details

1. Use Adam optimiser
2. The LSTM and linear layer are trained with learning rate of $1e-3$.
3. The pretrained Resnet50 is finetuned with learning rate of $1e-4$.
4. Training batch size is 200.
5. All frames are resized (224,224)
6. Images are sometimes randomly flipped (for data augmentation)



### Metrics

1. Overall accuracy: $78.4$%
2. Overall precision: $67.3$%
3. Overall recall: $58.6$%
4. Per phase recall and precision are as follows:

| Phase | Precision (%) | Recall (%) |
| ----- | ------------- | ---------- |
| BN    | 81.1          | 74.5       |
| BNU   | 95.4          | 98.5       |
| DVC   | 65.3          | 30.5       |
| LN    | 86.9          | 67.5       |
| NVB   | 68.0          | 80.7       |
| PA    | 66.3          | 52.7       |
| SVVD  | 75.6          | 64.6       |



### To Reproduce

1. First need to extract frames of all videos. Run:

   ```shell
   python convert_data_to_vdlp.py path/to/videos frames/save/path 1
   ```

2. Then compute metrics by running:

   ```shell
   python main.py --resume_from saved_models/baseline.pth saved_models/v2.pt  --only_test
   ```

   where frames/save/path is where frames are stored (same as the one in the first command)
