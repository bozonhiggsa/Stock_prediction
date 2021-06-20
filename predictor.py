import argparse
import settings


if __name__ == '__main__':
    # arguments parser for running directly from the command line
    parser = argparse.ArgumentParser(description='Parameters for training and prediction')
    parser.add_argument('--mode', '-m', help='Mode - train or predict', default='train')
    parser.add_argument('--company', '-c', help='Company Name', default='NFLX')
    args = parser.parse_args()

    # verify arguments from command line
    if args.mode not in settings.mode:
        raise Exception("Mode value is incorrect")
    if args.company not in settings.companies:
        raise Exception("Company name is incorrect")


