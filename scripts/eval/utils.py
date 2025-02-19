import seaborn as sns
import json
import os

def get_last_checkpoint(model_dir):
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)
    return trainer_state['max_steps']


def get_seaborn_palette(num_colors):    
    if num_colors <= 10:
        return sns.color_palette("Paired", num_colors)
    
    paired_palette = sns.color_palette("Paired", 10)

    extra_colors = [
        (0.6, 0.85, 0.8), (0.3, 0.6, 0.55),     # Light and dark seafoam
        (1.0, 0.8, 0.7), (0.85, 0.45, 0.3),     # Light peach and deep terracotta
        (0.8, 0.75, 0.9), (0.6, 0.5, 0.75),     # Light and dark lavender
    ]

    extended_palette = paired_palette + extra_colors[:num_colors-10]
    return extended_palette