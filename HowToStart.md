# HowTo Start Cheat Sheet

## Create python environment

```bash
micromamba create -n agentic python=3.11 -y
micromamba activate agentic
```

## Install deps

```bash
pip install -r requirements.txt jupyter
```

## Run the notebook:

You have two options:

### From the terminal:

jupyter notebook 01_reflection.ipynb

### From Cursor: 

With the agentic environment activated, just open 01_reflection.ipynb in Cursor and select the agentic kernel from the kernel picker (top-right of the notebook). If it doesn't appear, you may need to install ipykernel and register it:

```bash
pip install ipykernel
python -m ipykernel install --user --name agentic --display-name "Python (agentic)"
```

Then select "Python (agentic)" as the kernel in Cursor's notebook UI.