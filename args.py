from argparse import ArgumentParser
import sys

def get_args():
    parser = ArgumentParser(description="RoMe")
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--checkpoint', type=str, default='saved_model/')
    parser.add_argument('--no_tqdm', default=False, action='store_true', help='disable tqdm progress bar')
    parser.add_argument('--gammatype', type=str, default='static', choices=['static', 'len-norm', 'idx-norm'])
    parser.add_argument('--gammaval', type=float, default=0.07, help="when --gammatype is static set a value")
    parser.add_argument('--emdlm', type=str, default="albert-large-v2")
    parser.add_argument('--results_path', type=str, default='query_text')
    parser.add_argument('--deltated', type=float, default=0.65)
    parser.add_argument('--deltaemd', type=float, default=0.6)
    args = parser.parse_args()
    return args