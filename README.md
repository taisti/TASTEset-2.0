Repo for TASTEset-2.0.

So far Python 3.6.9 has been used. If you have other version, perhaps it is
fine if you just remove versions from `requirements.txt`. You can also consider 
using [pyenv](https://github.com/pyenv/pyenv).

Run
1) If you use virtualenv
```bash
python -m venv tasteset_env
. ./tasteset_env/bin/activate
pip install --upgrade pip  # upgrade pip to newest version
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2) If you use pyenv
```
pyenv virtualenv 3.6.9 tasteset_env
pyenv activate tasteset_env
pip install --upgrade pip  # upgrade pip to newest version
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To freely import all scripts, please add the followings to the PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path-to-tasteset-repo>/src
```