import argparse
import json

def convert_to_jsonl(src_path, out_path):
    with open(out_path, 'w') as out_file:
        for line in open(src_path, 'r'):
            line = line.split('\t')
            sentence = {'text': line[1].strip(), 'meta':{'sentence_id':line[0]}}
            out_file.write(json.dumps(sentence) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = args.input_path[:-3]+'jsonl'

    convert_to_jsonl(args.input_path, args.ouput_path)

if __name__ == '__main__':
    main()