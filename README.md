# End to End Machine Learning Pipeline

🚀 **Live Demo**: [https://end-to-end-machine-learning-pipeline-qskv.onrender.com/](https://end-to-end-machine-learning-pipeline-qskv.onrender.com/)



---

## 🧭 API Endpoints

| Endpoint       | Description                                           |
|----------------|-------------------------------------------------------|
| `/train`       | ✅ **Train the model** — must be run first before using `/predict`. |
| `/predict`     | 🔍 Submit input data and get predictions.             |
| `/home` or `/` | 🏠 Web interface to interact with the model.          |

> ⚠️ **Important**: Visit `/train` once after deployment to build and save the model.

---


## WorkFlows


1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py


## How to run ?
### STEPS:


clone the repository

```bash
https://github.com/Templar121/End-to-End-Machine-Learning-Pipeline
```

### STEP A - Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```

### OR
### STEP B - Create a venv environment

```bash
python -m venv mlproj
```

```bash
source mlproj/Scripts/activate
```


### STEP 02 - Install the Requirements

```bash
pip install -r requirements.txt
```


