from .gaussians_4dgs import GaussianModel as Gaussian4DGS


def get_model(pcd, radius, model_args, opt_args):
    if model_args.name == "4dgs":
        model = Gaussian4DGS(pcd, radius, **vars(model_args))
    else:
        print("Model name {} not found!".format(model_args.name))

    model.build_opt(opt_args)
    return model
