Workflow to setup python 3.12.2: 
# Install python 3.12.2
winget install Python.Python.3.12
# Verify installation
py --list
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name=venv-3.12.2 --display-name="Python 3.12.2 (Project)"

# Go to Visual Studio Code and select the Kernel that has been created