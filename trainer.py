import os
from pytorch_lightning import Trainer
from test_tube import Experiment
from lightning_model import CoolModel

model = CoolModel()
exp = Experiment(save_dir=os.getcwd())

# train on cpu using only 10% of the data (for demo purpose)
trainer = Trainer(experiment=exp, max_nb_epochs=1, train_percent_check=0.1)

# train on 4 gpus
# trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=[0, 1, 2, 3])

# train on 32 gpus across 4 nodes (make sure to submit appropriate SLURM job)
# trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=[0, 1, 2, 3, 4, 5, 6, 7], nb_gpu_nodes=4)

# train (1 epoch only here for demo)
trainer.fit(model)

# view tensorflow logs
print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
print('and going to http://localhost:6006 on your browser')
