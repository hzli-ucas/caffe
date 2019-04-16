The `crnn.caffemodel` here is converted from the `crnn_demo_model.t7` model offered by [bgshih](https://github.com/bgshih/crnn).

We use pytorch\*, the `load_lua` function to read the .t7 file and get a `hashable_uniq_dict`, it has two tensor lists named *parameters* and *bnVars*. Each tensor in the lists contains the weights/biases/means/vars of a convolution/linear/lstm/batchnorm layer. We convert the pytorch-tensor into numpy type, then write the weights to a caffemodel file. 

\* pytorch has removed the `load_lua` function from its new version. But you can still find the original .py file by searching "read_lua_file" or "load_lua" within pytorch repository, there are some commits including the file. For example, [this one](https://github.com/pytorch/pytorch/blob/c6529f4851bb8ac95f05d3f17dea178a0367aaee/torch/utils/serialization/read_lua_file.py).
