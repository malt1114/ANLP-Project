# ANLP-Project

This repository contains the code and instructions to reproduce the findings of our project.

## Steps to Reproduce

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/ANLP-Project.git
cd ANLP-Project
```

### 2. Reproduce Baseline

1. Navigate to the `notebooks` folder and open `Baseline.ipynb`.
2. Run the notebook from start to finish.
3. The baseline experiment results will be generated.

### 3. Reproduce Bi-LSTM

1. Train the models using the following scripts:
   - `model_train_easy.py`
   - `model_train_hard.py`

   **Note:** Training took approximately 8 hours on ITU's HPC.

2. After training, the models will be saved in the `models/` folder.
3. Manually copy the final model (or the model with the lowest validation loss) for each script to:
   - `models/easy` (for the easy script)
   - `models/hard` (for the hard script)

4. Modify the `notebooks/evaluation.ipynb` notebook to load the trained models:
   - Update all occurrences of the model paths in the notebook as shown below:

     ![image](https://github.com/user-attachments/assets/83a900f3-9f45-4c8b-9d3b-f4e31ebaa691)

5. Run the entire notebook to reproduce the experimental results.

### 4. Reproduce Analysis

1. Open the `notebooks/Analysis.ipynb` notebook.
2. Run the notebook from start to finish to reproduce the model analysis.

### 5. Reproduce Data Creation (Optional)

The typoglycemia dataset used in this repository is already provided. If you want to regenerate it:

1. Run the `run_preprocessing.py` script:

   ```bash
   python run_preprocessing.py
   
