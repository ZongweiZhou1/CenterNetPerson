from .base import Base, load_cfg, load_nnet
from config import system_configs
from db.datasets import datasets
import pkg_resources
import importlib
import os

_package_name = __name__


def get_file_path(*paths):
    path = "/".join(paths)
    return pkg_resources.resource_filename(_package_name, path)


class CenterNet(Base):
    def __init__(self, cfg_file, iter=10000, suffix=None):
        from test.centernet import inference

        model = importlib.import_module('models.%s'%cfg_file).model
        if suffix is None:
            cfg_path = os.path.join(system_configs.config_dir, "%s.json" % cfg_file)
        else:
            cfg_path = os.path.join(system_configs.config_dir, "%s-%s.json" % (cfg_file, suffix))
        model_path = get_file_path("..", "cache", "nnet", cfg_file, "%s_%d.pkl" % (cfg_file, iter))
        cfg_sys, cfg_db = load_cfg(cfg_path)
        cfg_sys["snapshot_name"] = cfg_file
        system_configs.update_config(cfg_sys)
        dataset = system_configs.dataset
        train_split = system_configs.train_split
        val_split = system_configs.val_split
        test_split = system_configs.test_split

        split = {
			"training": train_split,
			"validation": val_split,
			"testing": test_split
		}["validation"]

        demo = datasets[dataset](cfg_db, split)

        centernet = load_nnet(demo)
        super(CenterNet, self).__init__(demo, centernet, inference, model=model_path)


