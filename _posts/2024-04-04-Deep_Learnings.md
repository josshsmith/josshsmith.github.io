# Deep Learnings
Put a table of contents in here  

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


 

  
