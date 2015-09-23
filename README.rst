This has moved to lasagne
=========================

**See https://github.com/skaae/recurrent-spatial-transformer-code**   

The SpatialTransformerLayer has been moved to Lasagne. The Lasagne version is
identical except for very minor interface changes.

See <http://lasagne.readthedocs.org/en/latest/modules/layers/special.html#lasagne.layers.TransformerLayer>.
You need to install lasagne from github to get the spatial transformer layer.

I created an example usage here <https://github.com/Lasagne/Recipes/pull/7>.
It's currently a pull request.

Implementation of Spatial Transformer Networks[1]
=======
Lasagne implementation of Spatial Transformer Networks

You can import ``TransformerLayer`` from transformerlayer.py and use it as any
other lasagne layer.

.. image:: https://raw.githubusercontent.com/skaae/transformer_network/master/combined_small.png
    :alt: Example
    :width: 200
    :height: 140
    :align: center


Please cite this repository if you use the code.

References
=======

[1] Jaderberg, Max, et al. "Spatial Transformer Networks." arXiv preprint arXiv:1506.02025 (2015).
