
def _get_scope_dict():
    from . import activations, callbacks, grud_layers, layers

    merge_dict = lambda x, y: dict(list(x.items()) + list(y.items()))
    scope_dict = {}
    scope_dict = merge_dict(scope_dict, activations._get_activations_scope_dict())
    scope_dict = merge_dict(scope_dict, callbacks._get_callbacks_scope_dict())
    scope_dict = merge_dict(scope_dict, grud_layers._get_grud_layers_scope_dict())
    scope_dict = merge_dict(scope_dict, layers._get_layers_scope_dict())
    return scope_dict
