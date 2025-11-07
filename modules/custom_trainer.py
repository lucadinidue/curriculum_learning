from transformers import Trainer, TrainerCallback
from torch.utils.data import SequentialSampler
import torch


#class NoShuffleTrainer(Trainer):
#    def _get_train_sampler(self) -> torch.utils.data.Sampler:
#        return SequentialSampler(self.train_dataset)


# Changed for updated transformers version
class NoShuffleTrainer(Trainer):
    def _get_train_sampler(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Sampler:
        return SequentialSampler(dataset)


class IncrementalStepsCheckpointCallback(TrainerCallback):
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        control.should_save = False

        if step <= 4000 and step % 400 == 0:
            control.should_save = True

        elif step > 4000 and step % 4000 == 0:
            control.should_save = True

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
