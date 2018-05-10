# doggo-cv-chatbot
Doggo Breed Detection Chatbot Project:

Data Science Process:

Gathering dataset
1. Scrape relevant website for all breeds of dogs and save to csv [DONE]
2. Download images of each class (breed) using Google CustomSearch API and requests [DONE]

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
3. Histogram equalisation to increase contrast of low contrast image by balancing out distribution of pixel densities [To-do]

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
4. Apply transfer learning using Google pretrained image recognition model [To-do]

Validation: [To-do]
1. Test set from downloaded images (20%)
  - Classification accuracy
2. Photo images

--

Chatbot Development and Integration Process (TBC):

