# doggo-cv-chatbot
<b>Doggo Breed Detection Chatbot Project: </b>
<br/>
<br/>
<br/>

Data Science Process:<br/>
<br/>

Update 09/07/2018:
1. Downloaded ~90-100 images per class (max limit of 100 search results per search from Google Search)
	- Resized images to 250x250 upon completion of download for each image to reduce HD memory usage
	- Total of 19,778 images downloaded, 654 MB total
2. Performed transfer learning on Inception V3 image classification model, added a FC layer and model was trained for ~22 epochs using data augmentation via Keras Image Datagen
	- Tested on unseen dog images from Google search
	  1. 10 images per breed for top 5 most popular dog breeds and top 5 least popular breeds (~70% accuracy)
	- Tested on unseen dog images of friends' dogs
	  1. total of 63 images consisting of a poodle, a shiba inu and a schnauzer (73% accuracy)
3. Moving on to developing Telegram Bot and integrating model's function with bot
4. Possible improvements to model after Telegram Bot development and integration
	- Histogram equalisation of images
	- Build model with 2 FC layer and retrain with existing train dataset
	- Train logistic regression model using Inception V3 bottleneck features (Ensemble)

Update 15/05/2018: (Updated plan)
1. Severe lack of instances for each class (30~40)
2. Download a minimum of 200 images per class (209 classes), scaling them down to 300x300 pixels
3. Perform 10 data augmentations per image, resulting in 2000 augmented images per class
4. Obtain bottleneck features of augmented images (2000*209) using InceptionV3, saving it in a .npy file
5. Train nn model using bottleneck features on 2 FC NN, then 3 FC NN; both with dropout of 0.5 to reduce overfitting
6. Train logistic regression model using bottleneck features, setting to 100 iterations
7. If validation accuracy >= 0.5 then fine tune last convolution block of InceptionV3 by making it trainable<br/>
<br/>

Gathering dataset
1. Scrape relevant website for all breeds of dogs and save to csv [DONE]
2. Download images of each class (breed) using Google CustomSearch API and requests [DONE] <br/>
<br/>

Preprocessing data 
1. Build image preprocessing module with ImageProcessor class that:
    - Checks images in a directory, filtering out images that are unprocessable by opencv [DONE]
    - Build encoding and decoding dictionaries for labels for neural net input and output respectively [DONE]
    - Resize images to specified size otherwise defaults to 200x200 pixels: [DONE]
      1. Maintains aspect ratio and scales image down to 200 pixels for its larger edge
      2. Pads the shorter edge if pad == True
    - Normalises images on channel level using standard scaling [DONE]
    - Loads images from a directory, matching images to their labels via their filenames and returns: [DONE]
      1. 4D Numpy array of (number of examples, image height, image width, number of channels)
      2. 1D Numpy array of labels in the same order as the loaded images
2. Checked through download images for non-dog breed images and removed them [DONE]
3. Histogram equalisation to increase contrast of low contrast image by balancing out distribution of pixel densities [To-do]<br/>
<br/>

Modeling
1. Develop a simple 3 layer CNN model with pooling using Keras (model v1): [DONE]
    - Kernel size of 3x3
    - Pooling size of 3x3
    - Dropout of 0.5 (Prevent overfitting)
2. Decide on an optimiser to use [DONE]
    - Tried out SGD as optimiser but decided on Nadam (Nesterov momentum adaptive moment estimation)
      - Nesterov momentum in place of standard momentum used in adam optimiser
      - Uses weighted average of past n gradients to calculate parameter (weight) update
3. Train model using simple CNN architecture: [In-progress]
    - Tried SGD but did not normalise images nor perform data augmentation (End April 2018)
      1. Did not work so well, normalising features is important for nn to converge to a local/global optima as it standardise the inputs
      2. Low number of observations per class (~30-40 images per class) 
    - Tried Nadam and Adadelta on image-level normalised images - per channel on image level (Early May 2018)
      1. Seems to converge but speed is an issue (Half a day to reach 0.05 for validation accuracy)
      2. Used ImageDataGenerator from keras to augment training dataset on the flow to specifications:
        - Almost 'unlimited' training examples
    - Trying out Nadam on dataset-level normalised images - per channel on dataset level [In-progress]
      1. Increase number of filters for each convolution layer by a multiple of 2 of the previous layer's (slows down training)
      2. Added 1 more fully connected layer before softmax output and increase number of nodes to be between number of output nodes and number of training examples (slows down training)
      3. Increase max pooling kernel size from 2x2 to 3x3 to (speed up training)
      4. Nadam on learning rate of 0.03 (speed up training)
4. Apply transfer learning using Inception V3 image recognition model [DONE]
	- Sticking to model built with transfer leanring due to the high computational cost of building a nn image classifier using a laptop
	- Added a FC layer and set existing pre-trained layers to be untrainable
	- Set FC layer to use RMSProp<br/>
<br/>

Validation: [DONE]
1. Validation sets
    - 10 dog breeds images from Google (10 per breed): ~70% accuracy
	- 63 dog images of friends' dogs (poodle, shiba inu, schnauzer): ~73% accuracy
<br/>

Post-model training and validation: 
1. Build pre and post processing pipeline [DONE]
    1. Preprocess input image
        - Check processability [To-do]
        - Resizing (OpenCV resize)
        - Normalisation using Inception V3 preprocess
    2. Postprocess output
        - Decode output to class labels
2. Further modularise image_processing module [To-do]<br/>
<br/>
<br/>

--

Chatbot Development and Integration Process (TBC):

