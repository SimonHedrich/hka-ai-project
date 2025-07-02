import importlib

def instantiate_from_target(cfg):
    target_path = getattr(cfg, "target", None)
    if not target_path:
        raise ValueError("Config has no target attribute")

    module_path, class_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    cfg_dict = cfg.model_dump(exclude={"target"})

    if "optimizer" in cfg_dict:
        opt = getattr(cfg, "optimizer")
        if hasattr(opt, "model_dump"):
            raise RuntimeError("Optimizer is still a config")
        else:
            cfg_dict["optimizer"] = opt

    return cls(**cfg_dict)

