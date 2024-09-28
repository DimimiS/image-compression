
# Image Compression With The Use Of Neural Networks

This is a repo for my diploma thesis.

## How to run

You will need to download the requirements existing in requirements.txt by running the command:

```python
pip install -r requirements.txt
```

There three possible options to run the file: train, compress, decompress.
If you want to display more information during the process, you should include the --verbose argument.

For instance:

```command
python --verbose model.py your_run_choice
```

### Train
There are several arguments you can pass for training. To list them:
- lambda: lambda value
- train_glob: custom dataset
- num_filters: the middle layer number filters
- train_path: path to save model
- batchsize: size of training batch
- patchsize: size of images for training and validation
- epochs: max number of epochs
- steps_per_epoch: steps to be executed for each epoch
- max_validation_steps: limit to validation steps
- preprocess_threads: policy for mixed precision
- check_numerics: check for NaN and Inf

Each of these arguments can by used like this:

```command
python model.py train --argument argument_value
```

### Compress And Decompress
Similarly, there a few arguments that can be used:
- input_file: Required -> file to be processed
- output_file: Optional -> final name of processed file

The command should look like this for the compress command:

```command
python model.py compress input_file_name --output_file output_file_name
```

And like this for the decompress command:
```command
python model.py decompress input_file_name --output_file output_file_name
```

#### Implementation progress
1. The first try for the creation of the model was quite simple and based on the Cifar10 dataset. 
2. Both binary crossentropy and mean square error were used. The results are very blurry and the small size of the images of the Cifar10 dataset are suspected.
3. Locally downloaded STL10 dataset and results were of better quality compared to Cifar10. Conclusion, image resolution is _critical_.
4. Initialization of code following paper research with the Imagenette Dataset.
5. Install the imagenette dataset provided by fastai.
6. Use educational purposes code from https://github.com/tensorflow/compression/blob/master/models/bls2017.py#L108.
7. Refactor educational code to fit desired model.
8. Evaluate and test the model.