from transformers import Trainer, TrainerCallback
from torch.utils.data import SequentialSampler
import torch


class NoShuffleTrainer(Trainer):
    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        return SequentialSampler(self.train_dataset)
    

class IncrementalStepsCheckpointCallback(TrainerCallback):
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        control.should_save = False

        if step <= 100000 and step % 10000 == 0:
            control.should_save = True

        elif step <= 1000000 and step % 100000 == 0:
            control.should_save = True        

        elif step > 1000000 and step % 1000000 == 0:
            control.should_save = True
