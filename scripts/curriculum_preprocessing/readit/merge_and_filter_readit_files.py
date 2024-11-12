import csv

def copy_sentences(src_path, csv_writer):
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        for row in csv_reader:
            scores = [float(score) for score in row[2:]]
            if not any(el > 100 for el in scores):
                csv_writer.writerow([row[0], row[1], row[-1]])

def main():
    src_path_1 = '../../../data/dataset_samples/sample_1_readit_first_pass.csv'
    src_path_2 = '../../../data/dataset_samples/sample_1_readit_second_pass.csv'
    out_path = '../../../data/dataset_samples/sample_1_readit_global.csv'

    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file)
        copy_sentences(src_path_1, csv_writer)
        copy_sentences(src_path_2, csv_writer)   



if __name__ == '__main__':
    main()