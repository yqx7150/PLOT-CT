# Training & Testing
## Training
1. To download AAPM training and testing data

2. Generate training pre-log sinogram by running the script in `PLOT-CT_utils` folder
```
cd PLOT-CT_utils
python create_pre_log.py
```

3. To pretrain DiffIR_S1
- Modify the **file path and training configurations** in `./options/train_DiffIRS1.yml`
- Run the training script
```
sh trainS1.sh
```

4. To train DiffIR_S2
- Modify the **file path and training configurations** in `./options/train_DiffIRS2.yml`
- Run the training script
```
sh trainS2.sh
```

**Note:** The above training script uses 1 GPUs by default.

## Testing
1. Generate testing pre-log sinogram by running the script in `PLOT-CT_utils` folder
```
cd PLOT-CT_utils
python create_pre_log_test_datasets.py
```

2. Modify the test configuration
- Adjust the **dataset path and test configurations** in `./options/test_DiffIRS2.yml`

3. Run the test script
```
sh test.sh
```
# Dependencies

```
basicsr == 1.4.2
torch == 2.4.1
numpy == 1.24.4
torchvision == 0.19.1
odl == 0.8.1
```