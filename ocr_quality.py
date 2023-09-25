import argparse
import pandas as pd
import Levenshtein as lev


def parse_arguments():
    parser = argparse.ArgumentParser(description='Measure precision, recall and F1-score of the OCR')
    parser.add_argument('-o', '--ocr_output', help='Plain text file with the OCR output')
    parser.add_argument('-gt', '--ground_truth', help='Plain text file from the Gold Standard')
    
    return parser.parse_args()

def measure_quality(ocr_output, ground_truth):
    matching_parts = lev.matching_blocks(lev.editops(ocr_output, ground_truth), ocr_output, ground_truth)
    true_pos = len(''.join([ocr_output[x[0]:x[0]+x[2]] for x in matching_parts]))
    
    precision = true_pos / len(ground_truth)
    recall = true_pos / len(ocr_output)
    f_score = 2 * ((precision * recall) / (precision + recall))
    
    return precision, recall, f_score

def main():
    args = parse_arguments()

    with open(args.ocr_output, mode='r', encoding='utf-8') as ocr_text:
        ocr_text = ocr_text.read()

    with open(args.ground_truth, mode='r', encoding='utf-8') as gold_text:
        gold_text = gold_text.read()

    precision, recall, f_score = measure_quality(ocr_text, gold_text)
    quality_data = pd.DataFrame(
        data={
            'Text': [args.ocr_output.split('/')[2]],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f_score]
        }
    )
    with open('sample_ocr/ocr_quality.csv', mode='a', encoding='utf-8') as measurements_file:
        print(quality_data.to_csv(index=False, header=False), file=measurements_file, end='')

if __name__ == '__main__':
    main()
