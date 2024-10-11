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
        
        if step <= 30 and step % 10 == 0:
            control.should_save = True

        elif step <= 90 and step % 30 == 0:
            control.should_save = True        

        elif step > 90 and step % 100 == 0:
            control.should_save = True
