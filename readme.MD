# Which files to upload github
`main.py` `requirements.txt`

# Create a virtual environment
`python -m venv venv`

# How to activate the virtual environment
`source ./venv/bin/activate` please deactivate if any conda environment is active

# Which packages to install
`pip install fastapi "uvicorn[standard]" xgboost joblib scikit-learn`

# How to create requirements.txt file
`pip freeze > requirements.txt`

# How to run the app in reload mode
`uvicorn main:app --host 0.0.0.0 --reload`