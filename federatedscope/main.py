import os
import sys

# 设置可见的GPU为4、5、6、7
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import (
    get_client_cls,
    get_server_cls,
)
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

# cfg_file = "/home/yfman/FedBiOT/federatedscope/llm/baseline/testcase.yaml"
cfg_file = "/home/yfman/FedBiOT/federatedscope/llm/baseline/llama_math.yaml"
# cfg_file = "/home/yfman/FedBiOT/federatedscope/llm/baseline/test.yaml"
opts = []
client_cfg_file = None
if __name__ == "__main__":
    init_cfg = global_cfg.clone()
    # args = parse_args()
    # if args.cfg_file:
    #     init_cfg.merge_from_file(args.cfg_file)
    if cfg_file:
        init_cfg.merge_from_file(cfg_file)
    # if cfg_file:
    #     init_cfg.merge_from_file(cfg_file)
    print("args.opts: ", opts)
    cfg_opt, client_cfg_opt = parse_client_cfg(opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    # print("args.client_cfg_file: ", args.client_cfg_file)
    if client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(client_cfg_file, "r"))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(), client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    if init_cfg.federate.client_idx_for_local_train != 0:
        init_cfg.federate.client_num = 1
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train]
        data = new_data

    init_cfg.freeze()

    runner = get_runner(
        data=data,
        server_class=get_server_cls(init_cfg),
        client_class=get_client_cls(init_cfg),
        config=init_cfg.clone(),
        client_configs=client_cfgs,
    )
    _ = runner.run()
