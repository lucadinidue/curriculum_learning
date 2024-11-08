import json
import os

def get_last_checkpoint(model_dir):
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)
    return trainer_state['max_steps']