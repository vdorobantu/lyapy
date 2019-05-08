

py -3 -m venv .venv
call .venv\Scripts\activate
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
py -3  -m ipykernel install --user --name .venv --display-name "Virtual Environment"
