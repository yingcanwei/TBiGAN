# TBiGAN
Adversarial Cycle Consistent Learning for Domain Adaption
This project uses a Transferabel Bidirectional Generative Adversarial Network with Cycle-Consistence Constrains to learn a mapping from source domain to target domain.

## Setup

### Prerequisites
- Tensorflow 1.4.1,Python 2.7 or above

### Recommended
- Linux with Tensorflow (GPU edition + cuDNN will be better)

### Using Pre-traied model that we have already trained before.
As tarin a model will take more than 48 hours depending on GPU, on CPU you will be waiting for a bit.
Therefore, we  provided an pre-trained model for you just for testing

### Getting Started
There are also links to pre-trained models alongside for edge2shoes dataset, note that the pre-trained models require the current version of edge2image.py

```sh
# clone this repo
git clone https://github.com/yingcanwei/TBiGAN.git
cd TBiGAN
```
Download the Pre-trained model: [edges2shoesModel.zip](https://drive.google.com/file/d/14WROFTs4unFS4PfOeSJw7v-YWWxMasao/view)
Unzip the pre-train model and then copy this 'edges2shoesModel' folder to 'edge2image' folder 

```sh
# test the model
python edge2image.py \
  --mode test \
  --output_dir edges2shoesDemo_result \
  --input_dir edges2shoesDemotest \
  --checkpoint edges2shoesModel
```

The test run will output an HTML file at `edges2shoes_result/index.html` that shows input/output/target image sets.
You could also check the original output picture file named fileame+ "-outputs.png" extension in `edges2shoes_result/images/` folder.  

### Getting Started from Scratch(Train Model by yourself) 

```sh
# clone this repo
git clone https://github.com/yingcanwei/edge2image.git
cd edge2image
# download the edges2shoes dataset (generated from http://vision.cs.utexas.edu/projects/finegrained/utzap50k/)
python tools/download-dataset.py edges2shoes
# train the model (this may take more than 48 hours depending on GPU, on CPU you will be waiting for a bit)
python edge2image.py \
  --mode train \
  --output_dir edges2shoes_train \
  --max_epochs 200 \
  --input_dir edges2shoes/train \
  --which_direction AtoB
# test the model
python edge2image.py \
  --mode test \
  --output_dir edges2shoes_test \
  --input_dir edges2shoes/val \
  --checkpoint edges2shoes_train
```

The test run will output an HTML file at `edges2shoes_test/index.html` that shows input/output/target image sets.

### Getting Started from any edge picture (Draw an edge picture by yourself) 
This part instruction is used to test an picture that draw by yourself.

```sh
# clone this repo
git clone https://github.com/yingcanwei/edge2image.git
cd edge2image
```
Download the Pre-trained model: [edges2shoesModel.zip](https://drive.google.com/file/d/14WROFTs4unFS4PfOeSJw7v-YWWxMasao/view)
Unzip the pre-train model and then copy this 'edges2shoesModel' folder to 'edge2image' folder 

Draw an edge picuture like delow:

<img src="doc/test1.png" width="256px"/>

And then copy your picture to the "doc" folder

Next, you need pre-process the picure draw by yourself to fit model input requirments as below:

```sh
＃Create necessary meta and output folder
mkdir ./resize
mkdir ./gray
mkdir ./combine

# pre-process the picture
python ./tools/process.py 
--operation all 
--pad --input_dir ./doc \
--resize_dir ./resize  \
--gray_dir ./gray \
--output_dir ./combine
```

The pre-processed file will be generate to folder named "combine" 

```sh
# test your picture
python edge2image.py \
  --mode test \
  --output_dir edges2shoesbyyourself_result \
  --input_dir combine \
  --checkpoint edges2shoesModel
```
The test run will output an HTML file at `edges2shoesbyyourself_result/index.html` that shows input/output/target image sets. Notice that if you draw the edge image, the target label in the result HTML file has no actual meaning, it is just for unification.
You could also check the original output picture file named fileame+ "-outputs.png" extension in `edges2shoesbyyourself_result/images/` folder.  
