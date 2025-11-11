<h2>TensorFlow-FlexUNet-Image-Segmentation-Aerial-Imagery-Road (2025/11/11)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Aerial Imagery Road</b> (Singleclass)  based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 572x572  pixels 
<a href="https://drive.google.com/file/d/1Ju3Fk5Jxgtzc-jnLMWNimEksJxD1Z8JQ/view?usp=sharing">
<b>Augmented-Road-ImageMask-Dataset.zip</b></a>
which was derived by us from 
<a href="https://gisstar.gsi.go.jp/gsi-dataset/02/H1-No17-572.zip">H1-No17-572.zip
</a> in Japanese web site <a href="https://gisstar.gsi.go.jp/gsi-dataset/02/index.html">GSI Dataset-02 (Roads)</a>
<br><br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>, 
our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as a second category. 
In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of <b>GSI Dataset-02 (Roads)</b>, which contains 2,000 images and overlay-masks respectively,
we used our offline augmentation tool <a href="https://github.com/sarah-antillia/ImageMask-Dataset-Offline-Augmentation-Tool"> 
ImageMask-Dataset-Offline-Augmentation-Tool</a> to augment the original dataset.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 572x572 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
Augmented dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/51.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/51.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/51.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/804.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/804.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/804.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/846.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/846.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/846.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from 
<a href="https://gisstar.gsi.go.jp/gsi-dataset/02/H1-No17-572.zip">
H1-No17-572.zip
</a>
in Japanese web site <a href="https://gisstar.gsi.go.jp/gsi-dataset/02/index.html">
GSI Dataset-02 (Roads)
</a> 
<br><br>
Please see also English page
<a href="https://www.gsi.go.jp/ENGLISH/index.html">
GSI: Geospatial Information Authority of Japan
</a>
<br>
<br>
<b>GSI Dataset-02 (Roads)</b><br>
<b>Overview</b><br>
This data is intended for use in machine learning, and is an 8-bit, 3-channel image of an aerial photograph taken with a 
ground pixel size of 20 cm, with pixels that show roads labeled in red (RGB:#FF0000). <br>
For use in machine learning, each piece of data consists of two pairs: the original image and the labeled image, 
and each pair can be identified by its file name.
<br><br>
<b>Image specifications</b><br>
Image sizes are available in two sizes: 572 x 572 pixels and 286 x 286 pixels. 
Both images have a bit depth of 8 bits per channel and are in PNG format.<br>
As of November 10, 2022, there are 2,000 pairs of 572 x 572 pixel images and 10,000 pairs of 286 x 286 pixel images available for download.
<br>
<br>
<b>Source</b><br>
 This data can be used under <a href="https://www.gsi.go.jp/ENGLISH/page_e30286.html">
 Geospatial Information Authority of Japan (GSI) Website Terms of Use</a>. <br>
 If you use it in a research presentation, etc., please indicate the source as follows:<br>
<b>
Geospatial Information Authority of Japan (2022): <br>
Training image data for road extraction using CNN, Geospatial Information Authority of Japan Technical Paper H1-No.17.
</b>
<br>
<br>
<b>License</b><br>
<a href="https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0">
Public Data License (Version 1.0)
</a>
<br>
<br>
<h3>
2 Road ImageMask Dataset
</h3>
<h4>2.1 Augmented ImageMask Dataset</h4>
 If you would like to train this Road Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1Ju3Fk5Jxgtzc-jnLMWNimEksJxD1Z8JQ/view?usp=sharing">
 <b>Augmented-Road-ImageMask-Dataset.zip </b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Road
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Road Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Road/Road_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<h4>2.2 Derivation colorized masks</h4>
The folder structure of the original H1-No17-572 dataset is the following.<br>
<pre>
 ./H1-No17-572    
  ├─org  (raw images)
  │  ├─1.png
    ...
  │  └─2000.png
  └─val  (mask overlay)
      ├─1.png
    ... 
      └─2000.png
</pre>
In our dataset, we generated each colorized mask from a pair of raw image and mask overy (mask-overlapped-image),
 by subtracting the raw image from the corresponding mask overlay, and colorizing the subtracted image with light-brown.<br> 
As shown below, the generated colorized masks are slightly different from the original solid red mask in mask overlay. 
<table>
<tr>
<th>mask overlay</th><th>raw image</th><th>colorized masks</th>
</tr>
<tr>

<td><img src="./projects/TensorFlowFlexUNet/Road/asset/overlay_3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/raw_3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/colorized_3.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/overlay_13.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/raw_13.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/colorized_13.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/overlay_700.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/raw_700.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/asset/colorized_700.png" width="320" height="auto"></td>
</tr>

</table>


<br><br>
<h4>2.3 Train images and masks sample</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Road TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Road/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Road and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Road 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
; road : BGR (80, 120, 255)
;                    road:light brown
rgb_map = {(0,0,0):0,(255,120,80):1, }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 15,16,17)</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 31,32,33)</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 33.<br><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/train_console_output_at_epoch33.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Road/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Road/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Road/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Road/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Road</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Road.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/evaluate_console_output_at_epoch33.png" width="720" height="auto">
<br><br>Image-Segmentation-Road

<a href="./projects/TensorFlowFlexUNet/Road/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Road/test was not low, anot dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1406
dice_coef_multiclass,0.919
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Road</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Road.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Road/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Road/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 572x572 pixels </b><br>

<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/34.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/302.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/302.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/302.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/370.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/370.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/370.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/520.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/520.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/520.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/945.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/945.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/945.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/images/832.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test/masks/832.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Road/mini_test_output/832.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>

<h3>
References
</h3>

<b>1. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>2. Satellite Imagery Road Segmentation</b><br>
Nithish<br>
<a href="https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812">
https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
</a>
<br>
<br>
<b>3. Deep Learning-based Road Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Road_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Road_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>4. A Comparative Study of Deep Learning Methods for Automated Road Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>


