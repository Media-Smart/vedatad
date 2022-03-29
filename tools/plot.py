import argparse

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot values from log file')
    parser.add_argument('logfile', help='log file path')
    parser.add_argument('--plot', '-p', help='variable to plot',
        choices=['loss', 'loss_segment', 'loss_cls', 'lr'])

    args = parser.parse_args()
    return args

def read_log(logfile):
    data = []
    with open(logfile, 'r') as log_file:
        for line in log_file:
            if '- INFO - Epoch' in line:
                data_point = {}
                line_split = line.split('] ')
                data_point['epoch'] = int(line_split[0].split('[')[1].strip(']'))
                data_point['iteration'], data_point['total_iterations'] = line_split[0].split('[')[2].split('/')
                
                for stat in line_split[1].split(', '):
                    k, v = stat.split(': ')
                    data_point[k] = float(v)

                data.append(data_point)

    return data


def plot(log_data, to_plot):
    plt.title(to_plot)
    plt.plot([x[to_plot] for x in log_data])
    plt.show()


def main():
    args = parse_args()
    log_data = read_log(args.logfile)
    plot(log_data, args.plot)


if __name__ == "__main__":
    main()