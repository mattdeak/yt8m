FROM ntaylor22/tensorflow-gpu

RUN pip install kaggle==1.3.11.1 && mkdir /root/.kaggle
COPY kaggle.json /root/.kaggle
COPY jupyter_notebook_config.json /etc/jupyter

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
