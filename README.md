## Attention-based Deep "Multiple Instance Learning" (MIL) turned into "multi-instance multi-label learning" (MIML)
The repo is a modified version of Jiawen Yao's [Atten_Deep_MIL](https://github.com/utayao/Atten_Deep_MIL) that is able to solve a MIML problem

The original code is based on ICML 2018 paper "Attention-based Deep Multiple Instance Learning" (https://arxiv.org/pdf/1802.04712.pdf)

In MIL each bag has only one label, where MIML has bags with miltiple labels.

So what examples can we think of that is MIML
- uploads of 1-n images tags the post with several tags. the MIML problem tries to determine what tag relates to what image
- You might have several images of a car damage, the car might have damages on multiple parts, with MIML you try to determine what part relates to what image

Good read about MIML is [Multi-Instance Multi-Label Learning withApplication to Scene Classification](https://papers.nips.cc/paper/3047-multi-instance-multi-label-learning-with-application-to-scene-classification.pdf)
The paper has this good representation of the different learning frameworks

![Alt text](images/supervised_frameworks.jpg)

I was not able to find any MIML image datasets so i desided to use [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and then randomly put 3-4 images in each bag

![Alt text](images/bag_0.jpg)

bag label = (car,dog,horse)

![Alt text](images/bag_1.jpg)

bag label = (cat,dog,horse,boat)

The challange is then to train the network only knowing the bag labels, and then get a network that will
1. predict the correct labels for a bag
2. predict what label in the bag each image belongs to

### Results from the implementation
The model's are trained for 100 epochs and the results is validated on a the test set


| | bag_accuracy | instance_accuracy | true_positive_accuracy |
| ------ |:----:|:----:|:---:|
|class 0 | 0.83 | 0.90 | 0.93|
|class 1 | 0.88 | 0.93 | 0.95|
|class 2 | 0.64 | 0.80 | 0.90|
|class 3 | 0.77 | 0.77 | 0.78|
|class 4 | 0.79 | 0.87 | 0.90|
|class 5 | 0.80 | 0.83 | 0.86|
|class 6 | 0.84 | 0.94 | 0.97|
|class 7 | 0.77 | 0.88 | 0.92|
|class 8 | 0.83 | 0.94 | 0.97|
|class 9 | 0.84 | 0.94 | 0.97|
|**MEAN** | **0.80** | **0.88** | **0.92**|

**bag_accuracy:** The number of bags containing a labe that is predicted to contain that label

**instance_accuracy:** How accurate the system was in selecting an image of a specific class, knowing that the bag is containing at least one image from the class

**true_positive_accuracy:** If the system predicted the bag to contain a class, how accurate it was to select the right image in the bag

### Reflections
I set out to try to solve this problem as i have a big dataset with labeled bags of images. I would like to train a traditional supervised image classifier, howewer that requires that i label each image in the image bags.

One option to was to start labeling at hand, but this is of cause time consuming and should be the last option if all else fails.

If these numbers translates, i would be more than happy to get ~~66%~~ 80% of my dataset sorted with a ~~90%~~ 92% accuracy.


### Future Work
I struggled quit a bit to get the output of the network to be multi-class, 
finally i got it to work by repeating the attention layer for each class, and concatenating the output before compiling the model.

As many each class in the multi class classification is quit unbalanced, as a image of a specific class is more oftent not in a bag.

Thanks to Dennis for providing a good [solution for applying class weight to multiclass](https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras),  A solution that also could be adopted to work with multiple input

Im still happy to give 12 homebrews to anyone that can improve the final layers of the network, still not sure my solution is the optimal :-)

![Alt text](images/homebrew.jpg)

### Run the code
Install requirements from requirements.txt and then run main
```console
pip install requirements.txt
python main.py
```


### Dataset
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)