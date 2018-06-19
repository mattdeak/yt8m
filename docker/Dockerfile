FROM ntaylor22/tensorflow-gpu

RUN pip install kaggle==1.3.11.1
COPY kaggle.json /home/.kaggle
COPY jupyter_notebook_config.json /etc/jupyter

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
