from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from utils import load_graph_data
from research_impl import DCRNNSupervisor
import numpy as np

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        # print(adj_mx)
        # print(adj_mx.shape)
        # print(type(adj_mx))
        # print(adj_mx.dtype)
        # x = supervisor_config['model']['num_nodes']
        # adj_mx = np.ones((x, x), dtype=np.float32)

        
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
