from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, set_seed, BertConfig, GPT2Config
import argparse
import json

def load_model_config(config_path, model_type):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        if model_type == 'bert':
            config =  BertConfig.from_dict(config)
        else:
            config = GPT2Config.from_dict(config)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, choices=['bert', 'gpt'])
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-c', '--config_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    set_seed(args.seed)

    config = load_model_config(args.config_path, args.model_type)
    if args.model_type == 'bert':
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_type == 'gpt':
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError("Unsupported model type. Choose 'bert' or 'gpt'.")
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
    