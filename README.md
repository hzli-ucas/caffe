This is the [caffe](https://github.com/BVLC/caffe) project for [**Reinterpreting CTC training as iterative fitting**. Hongzhu Li and Weiqiang Wang. Pattern Recognition, Volume 105, 2020.](https://arxiv.org/abs/1904.10619)

# Abstract

The connectionist temporal classification (CTC) enables end-to-end sequence learning by maximizing the probability of correctly recognizing sequences during training. The outputs of a CTC-trained model tend to form a series of spikes separated by strongly predicted blanks, know as the spiky problem. To figure out the reason for it, we reinterpret the CTC training process as an iterative fitting task that is based on frame-wise cross-entropy loss. It offers us an intuitive way to compare target probabilities with model outputs for each iteration, and explain how the model outputs gradually turns spiky. Inspired by it, we put forward two ways to modify the CTC training. The experiments demonstrate that our method can well solve the spiky problem and moreover, lead to faster convergence over various training settings. Beside this, the reinterpretation of CTC, as a brand new perspective, may be potentially useful in other situations. The code is publicly available at https://github.com/hzli-ucas/caffe/tree/ctc.

# Instructions

The custom implementation is based on the [windows branch of caffe](https://github.com/BVLC/caffe/tree/windows).

 - You can either clone the code with `git clone -b ctc https://github.com/hzli-ucas/caffe.git`, then directly build the project following [BVLC/caffe/windows](https://github.com/BVLC/caffe/tree/windows).
 - Or you can add the ctc implementaion into another caffe version, and all the related modifications should be made are listed below.

## Modifications

What we have done includes:

 - Convert the label type from `int` to `vector<int>`, so that caffe can deal with sequential label. ([commit e81e1ce](https://github.com/BVLC/caffe/commit/e81e1ce))
 - Add new layers [`ctc_loss_layer`](https://github.com/hzli-ucas/caffe/blob/ctc/include/caffe/layers/ctc_loss_layer.hpp), [`ctc_decoder_layer`](https://github.com/hzli-ucas/caffe/blob/ctc/include/caffe/layers/ctc_decoder_layer.hpp) and [`sequence_accuracy_layer`](https://github.com/hzli-ucas/caffe/blob/ctc/include/caffe/layers/sequence_accuracy_layer.hpp) to implement CTC.
 - Add layer [`lmdb_data_layer`](https://github.com/hzli-ucas/caffe/blob/ctc/include/caffe/layers/lmdb_data_layer.hpp) that reads from dataset created by [bgshih/crnn](https://github.com/bgshih/crnn).
 - Add layers that can cooperate the recurrent layer. ([commit bfb3ec2](https://github.com/BVLC/caffe/commit/bfb3ec2))

You can merge the CTC implementation into your caffe project through the following steps:

#### Here are the new files, put them to your own corresponding paths

[`include/caffe/layers`](https://github.com/hzli-ucas/caffe/tree/ctc/include/caffe/layers): `ctc_loss_layer.hpp`, `ctc_decoder_layer.hpp`, `sequence_accuracy_layer.hpp`, `lmdb_data_layer.hpp`, `reverse_layer.hpp`, `conv_to_recur_layer.hpp`.

[`src/caffe/layers`](https://github.com/hzli-ucas/caffe/tree/ctc/src/caffe/layers): `ctc_loss_layer.cpp`, `ctc_loss_layer.cu`, `ctc_decoder_layer.cpp`, `sequence_accuracy_layer.cpp`, `lmdb_data_layer.cpp`, `reverse_layer.cpp`, `reverse_layer.cu`, `conv_to_recur_layer.cpp`, `conv_to_recur_layer.cu`.

#### Here are the modified files, either modify your own files according to commit [e81e1ce](https://github.com/BVLC/caffe/commit/e81e1ce) and [bfb3ec2](https://github.com/BVLC/caffe/commit/bfb3ec2), or simply replace your original files with them.

[`include/caffe/util`](https://github.com/hzli-ucas/caffe/tree/ctc/include/caffe/util): `io.hpp`, `math_functions.hpp`, `db.hpp`, `db_leveldb.hpp`, `db_lmdb.hpp`.

[`src/caffe/util`](https://github.com/hzli-ucas/caffe/tree/ctc/src/caffe/util): `io.cpp`, `math_functions.cu`.

[`src/caffe/layers`](https://github.com/hzli-ucas/caffe/tree/ctc/src/caffe/layers): `data_layer.cpp`, `memory_data_layer.cpp`.

[`examples`](https://github.com/hzli-ucas/caffe/tree/ctc/examples): `cifar10/convert_cifar_data.cpp`, `mnist/convert_mnist_data.cpp`, `siamese/convert_mnist_siamese_data.cpp`.

#### Modify your proto file

Referring to [`src/caffe/proto/caffe.proto`](https://github.com/hzli-ucas/caffe/blob/ctc/src/caffe/proto/caffe.proto),
 - add three new messages `CTCParameter`, `ReverseParameter` and `LmdbDataParameter`,
 - modify message `Datum` by changing the attribute of `label` from 'optional' to 'repeated',
 - modify message `DataParameter` by adding `label_size`,
 - modify message `RecurrentParameter` by adding `time_steps`.

#### And finally, rebuild your caffe project.

## Usage

The prototxt files of network and solver are given in [examples/\_ctc](https://github.com/hzli-ucas/caffe/tree/ctc/examples/_ctc). After successfully building the project, you can run `train.bat` to train a new model.

The simulation tool mentioned in the paper is given as [sim_tool.py](https://github.com/hzli-ucas/caffe/blob/ctc/examples/_ctc/sim_tool.py).

#### Dataset

We use the same training set as [bgshih/crnn](https://github.com/bgshih/crnn), the code used to generate the dataset is [bgshih/crnn/tool/create_dataset.py](https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py).

The test set and Synth5k can be found at [liuhu-bigeye/enctc.crnn](https://github.com/liuhu-bigeye/enctc.crnn).
