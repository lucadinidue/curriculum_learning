from transformers import AutoModelForMaskedLM, set_seed,  BertConfig
import argparse
import random
import json
import os

def load_model_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        config =  BertConfig.from_dict(config)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-c', '--config_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()


    if args.seed is None:
        args.seed = random.randint(1, 1000)
        print(args.seed)
        args.output_path += f'_{args.seed}'
        if os.path.exists(args.output_path):
            exit(0)
    set_seed(args.seed)

    config = load_model_config(args.config_path)
    model = AutoModelForMaskedLM.from_config(config)
    model.save_pretrained(args.output_path)


    # model = AutoModelForMaskedLM.from_pretrained(model_name)
    # print(model.config)
    # model_name = 'google-bert/bert-base-uncased'
    # save_dir = os.path.join('local_models', model_name.split('/')[-1])
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    main()
    
