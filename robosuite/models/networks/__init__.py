from robosuite.models.networks.mlp import mlp


def get_network(type, **kwargs):
    if type == 'mlp':
        return mlp(**kwargs)

    return None