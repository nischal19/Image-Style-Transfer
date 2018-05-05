# Image-Style-Transfer

This repository contains implementation of Neural/image Style Transfer based on paper ["A Neural Algorithm of Artistic Style" by Gatys et. al.](https://arxiv.org/abs/1508.06576)

Apart from the content loss and style loss as mentioned in the paper a new loss called "Continuity loss" is introduced by us in the implementation. This loss accounted for the differences on consecutive frames in the stylised image when working on a gif.

## Running - 

python3 neural_style_red.py --content_folder <content_folder> --style <style_file> --output_folder <output_folder>

<content_folder> must contain the each frame in order of the gif 
<output_folder> is the destination where the stylized frames are stored

A gif file named "output.gif" is also stored at the location of the "neural_style_red.py" file folder

For obtaining the results for a single image the following command can be used :

python3 neural_style1.py --content <content_file> --styles <style_file> --output <output_file>

Use --iterations to change the number of iterations
Use --content-weight to change the weight of content of content image in the output image
Use --style-weight to change the weight of style of style image in the output image

## Requirements - 

### Data Files
 A pretrained VGG19 network is put in the the top level of this repository, or specify its location using the --network option.

### Dependencies
 The packages required for the code to work are :
 1. Tensorflow
 2. NumPy
 3. SciPy
 4. Pillow
 5. Imageio
