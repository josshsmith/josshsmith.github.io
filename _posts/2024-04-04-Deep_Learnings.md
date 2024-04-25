# Deep Learnings
1. TOC
{:toc}

## Deep Learning for *Coders*
We have gone from not being able to figure out if a certain animal is in an image to being able to do just that in a few minutes.
Recall that images are made of numbers, coloured images are in terms of red, green and blue.


This can be done very quickly and easy in python. For example,
```python
dls = DataBlock(
  blocks=(ImageBlock, CategoryBlock),
  get_items = get_image_files,
  splitter = RandomSplitter(valid_pct=0.2, seed=42,
  get_y=parent_label,
  item_tfms=[Resize(192, method = 'squish')]
).dataloaders(path)
```
DataBlocks give fastai all the information it needs to create a computer vision model.  
The example above has a bunch of preset data I will summarise:
* **blocks:** The kind of data we are using
* **get_items:** A function that retrieves the image files from a folder
* **splitter:** Defines how we split our data between training and testing data.
* **get_y:** A function to get the associated label from an image
* **item_tfms:** Resizing the image so they are all the same size


 ## How to use a Datablock and Datablock.dataloader
 You can show some of the data in a dataloader by doing the following:
 ```python
dls.show_batch(max_n=6)
```
This produces the following image:
<img width="727" alt="image" src="https://github.com/josshsmith/josshsmith.github.io/assets/141536363/2d12ffbc-6d3e-452f-813a-9d7e7c1996da">
This shows our images with their associated label.
We can use duckduckgo to retrieve images online and they will be mostly accurate.
**Creating a learning model**  
We want to create a model that can make predictions for what class an image belongs to. You can create a Learner, this is a class given by fastai that is capable of being fine tuned and making predictions based on unseen data. This is seen in the following code:  
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
This will create a vision learner using the resnet18 architecture. Resnet18 has 18 layers. You can use the following functions:
```python
learn.get_predict()
learn.predict()
```
These will return the models predictions for the data in the Datablock, or if using the predict function, you can pass new data to be predicted.
  
