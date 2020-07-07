!git clone https://github.com/matterport/Mask_RCNN.git

!python setup.py install

!pip show mask-rcnn

!git clone https://github.com/experiencor/kangaroo.git

!pip install tensorflow==1.15
import tensorflow
print(tensorflow.__version__)

!pip install keras==2.2.5
import keras
print(keras.__version__)

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
class CarDataset(Dataset):
  def load_dataset(self,dataset_dir,is_train=True):
    self.add_class("dataset",1,"car")
    self.add_class("dataset",2,"scratch")
    images_dir=dataset_dir+'/carimg/'
    annotations_dir=dataset_dir+'/maskimg/'
    for filename in listdir(images_dir):
      image_id=filename[:-4]
      if(is_train and int(image_id)>=75):
        continue
      if not is_train and int(image_id)<75:
        continue
      img_path=images_dir+filename
      ann_path=annotations_dir+image_id+'.xml'
      self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)
  def extract_boxes(self,filename):
    tree=ElementTree.parse(filename)
    root=tree.getroot()
    boxes=list()
    names=list()
    ids=list()
    for box in root.findall('.//object'):
      xmin=int(box.find('xmin').text)
      ymin=int(box.find('ymin').text)
      xmax=int(box.find('xmax').text)
      ymax=int(box.find('ymax').text)
      name=str(root.find('.//object/name').text)
      if(name=='car'):
        ids.append(1)
      else:
        ids.append(2)
      coors=[xmin,ymin,xmax,ymax]
      boxes.append(coors)
      names.append(name)
    width=int(root.find('.//size/width').text)
    height=int(root.find('.//size/height').text)
    return boxes,width,height,names,ids
  def load_mask(self,image_id):
    info=self.image_info[image_id]
    path=info['annotation']
    boxes,w,h,names,ids=self.extract_boxes(path)
    masks=zeros([h,w,len(boxes)],dtype='uint8')
    class_ids=list()
    for i in range(len(boxes)):
      box=boxes[i]
      row_s,row_e=box[1],box[3]
      col_s,col_e=box[0],box[2]
      masks[row_s:row_e,col_s:col_e,i]=int(ids[i])
      class_ids.append(self.class_names.index(names[i]))
    return masks,asarray(class_ids,dtype='int32')
  def image_reference(self,image_id):
    info=self.image_info[image_id]
    return info['path']
train_set=CarDataset()
train_set.load_dataset('car2',is_train=True)
train_set.prepare()
print('Train: %d'%len(train_set.image_ids))
test_set=CarDataset()
test_set.load_dataset('car2',is_train=False)
test_set.prepare()
print('Test: %d'%len(test_set.image_ids))

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
class CarDataset(Dataset):
  def load_dataset(self,dataset_dir,is_train=True):
    self.add_class("dataset",1,"car")
    self.add_class("dataset",2,"scratch")
    images_dir=dataset_dir+'/carimg/'
    annotations_dir=dataset_dir+'/maskimg/'
    for filename in listdir(images_dir):
      image_id=filename[:-4]
      if(is_train and int(image_id)>=75):
        continue
      if not is_train and int(image_id)<75:
        continue
      img_path=images_dir+filename
      ann_path=annotations_dir+image_id+'.xml'
      self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)
  def extract_boxes(self,filename):
    tree=ElementTree.parse(filename)
    root=tree.getroot()
    boxes=list()
    names=list()
    ids=list()
    for box in root.findall('.//object'):
      xmin=int(box.find('.//bndbox/xmin').text)
      ymin=int(box.find('.//bndbox/ymin').text)
      xmax=int(box.find('.//bndbox/xmax').text)
      ymax=int(box.find('.//bndbox/ymax').text)
      name=str(box.find('name').text)
      print(xmin)
      coors=[xmin,ymin,xmax,ymax]
      boxes.append(coors)
      names.append(name)
    width=int(root.find('.//size/width').text)
    height=int(root.find('.//size/height').text)
    return boxes,width,height,names,ids
  def load_mask(self,image_id):
    info=self.image_info[image_id]
    path=info['annotation']
    boxes,w,h,names,ids=self.extract_boxes(path)
    masks=zeros([h,w,len(boxes)],dtype='uint8')
    class_ids=list()
    for i in range(len(boxes)):
      box=boxes[i]
      row_s,row_e=box[1],box[3]
      col_s,col_e=box[0],box[2]
      masks[row_s:row_e,col_s:col_e,i]=1#int(ids[i])
      class_ids.append(self.class_names.index(names[i]))
    return masks,asarray(class_ids,dtype='int32')
  def image_reference(self,image_id):
    info=self.image_info[image_id]
    return info['path']
train_set=CarDataset()
train_set.load_dataset('car2',is_train=True)
train_set.prepare()
image_id=40
image=train_set.load_image(image_id)
mask,class_ids=train_set.load_mask(image_id)
bbox=extract_bboxes(mask)
display_instances(image,bbox,mask,class_ids,train_set.class_names)

'''from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
class CarDataset(Dataset):
  def load_dataset(self,dataset_dir,is_train=True):
    self.add_class("dataset",1,"car")
    self.add_class("dataset",2,"scratch")
    images_dir=dataset_dir+'/carimg/'
    annotations_dir=dataset_dir+'/maskimg/'
    for filename in listdir(images_dir):
      image_id=filename[:-4]
      if(is_train and int(image_id)>=75):
        continue
      if not is_train and int(image_id)<75:
        continue
      img_path=images_dir+filename
      ann_path=annotations_dir+image_id+'.xml'
      self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)
  def extract_boxes(self,filename):
    tree=ElementTree.parse(filename)
    root=tree.getroot()
    boxes=list()
    names=list()
    ids=list()
    for box in root.findall('.//object'):
      xmin=int(box.find('.//bndbox/xmin').text)
      ymin=int(box.find('.//bndbox/ymin').text)
      xmax=int(box.find('.//bndbox/xmax').text)
      ymax=int(box.find('.//bndbox/ymax').text)
      name=str(box.find('name').text)
      print(xmin)
      coors=[xmin,ymin,xmax,ymax]
      boxes.append(coors)
      names.append(name)
    width=int(root.find('.//size/width').text)
    height=int(root.find('.//size/height').text)
    return boxes,width,height,names,ids
  def load_mask(self,image_id):
    info=self.image_info[image_id]
    path=info['annotation']
    boxes,w,h,names,ids=self.extract_boxes(path)
    masks=zeros([h,w,len(boxes)],dtype='uint8')
    class_ids=list()
    for i in range(len(boxes)):
      box=boxes[i]
      row_s,row_e=box[1],box[3]
      col_s,col_e=box[0],box[2]
      masks[row_s:row_e,col_s:col_e,i]=1#int(ids[i])
      class_ids.append(self.class_names.index(names[i]))
    return masks,asarray(class_ids,dtype='int32')
  def image_reference(self,image_id):
    info=self.image_info[image_id]
    return info['path']
class PersonConfig(Config):
  NAME="car2_cfg"
  NUM_CLASSES=1+2
  STEPS_PER_EPOCH=12
train_set=CarDataset()
train_set.load_dataset('car2',is_train=True)
train_set.prepare()
print('Train: %d'%len(train_set.image_ids))
test_set=CarDataset()
test_set.load_dataset('car2',is_train=False)
test_set.prepare()
print('Test: %d'%len(test_set.image_ids))
config=PersonConfig()
config.display()
model=MaskRCNN(mode='training',model_dir='./',config=config)
model.load_weights('/content/drive/My Drive/Mask_RCNN/mask_rcnn_coco.h5',by_name=True,
                   exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
model.train(train_set,test_set,learning_rate=config.LEARNING_RATE,epochs=12,layers='heads')'''

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
class PersonDataset(Dataset):
  def load_dataset(self,dataset_dir,is_train=True):
    self.add_class("dataset",1,"car")
    self.add_class("dataset",2,"scratch")
    images_dir=dataset_dir+'/carimg/'
    annotations_dir=dataset_dir+'/maskimg/'
    for filename in listdir(images_dir):
      image_id=filename[:-4]
      if(is_train and int(image_id)>=75):
        continue
      if not is_train and int(image_id)<75:
        continue
      img_path=images_dir+filename
      ann_path=annotations_dir+image_id+'.xml'
      self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)
  def extract_boxes(self,filename):
    tree=ElementTree.parse(filename)
    root=tree.getroot()
    boxes=list()
    names=list()
    ids=list()
    for box in root.findall('.//object'):
      xmin=int(box.find('.//bndbox/xmin').text)
      ymin=int(box.find('.//bndbox/ymin').text)
      xmax=int(box.find('.//bndbox/xmax').text)
      ymax=int(box.find('.//bndbox/ymax').text)
      name=str(box.find('name').text)
      print(xmin)
      coors=[xmin,ymin,xmax,ymax]
      boxes.append(coors)
      names.append(name)
    width=int(root.find('.//size/width').text)
    height=int(root.find('.//size/height').text)
    return boxes,width,height,names,ids
  def load_mask(self,image_id):
    info=self.image_info[image_id]
    path=info['annotation']
    boxes,w,h,names,ids=self.extract_boxes(path)
    masks=zeros([h,w,len(boxes)],dtype='uint8')
    class_ids=list()
    for i in range(len(boxes)):
      box=boxes[i]
      row_s,row_e=box[1],box[3]
      col_s,col_e=box[0],box[2]
      masks[row_s:row_e,col_s:col_e,i]=1#int(ids[i])
      class_ids.append(self.class_names.index(names[i]))
    return masks,asarray(class_ids,dtype='int32')
  def image_reference(self,image_id):
    info=self.image_info[image_id]
    return info['path']
class PredictionConfig(Config):
  NAME="car2_cfg"
  NUM_CLASSES=1+2
  GPU_COUNT=1
  IMAGES_PER_GPU=1
def evaluate_model(dataset,model,cfg):
  APs=list()
  for image_id in dataset.image_ids:
    image,image_meta,gt_class_id,gt_bbox,gt_mask=load_image_gt(dataset,cfg,image_id,use_mini_mask=False)
    scaled_image=mold_image(image,cfg)
    sample=expand_dims(scaled_image,0)
    yhat=model.detect(sample, verbose=0)
    r=yhat[0]
    AP, _, _, _=compute_ap(gt_bbox,gt_class_id,gt_mask,r["rois"],r["class_ids"],r["scores"],r['masks'])
    APs.append(AP)
  mAP=mean(APs)
  return mAP
train_set=PersonDataset()
train_set.load_dataset('car2',is_train=True)
train_set.prepare()
print("Train: %d"%len(train_set.image_ids))
test_set=PersonDataset()
test_set.load_dataset('car2',is_train=False)
test_set.prepare()
print("Test: %d"%len(test_set.image_ids))
cfg=PredictionConfig()
model=MaskRCNN(mode='inference',model_dir='./',config=cfg)
model.load_weights('/content/drive/My Drive/Mask_RCNN/car2_cfg20200629T1007/mask_rcnn_car2_cfg_0001.h5',
                   by_name=True)
train_mAP=evaluate_model(train_set,model,cfg)
print("Train mAP: %.3f"%train_mAP)
test_mAP=evaluate_model(test_set,model,cfg)
print("Test mAP: %.3f"%test_mAP)

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
class PersonDataset(Dataset):
  def load_dataset(self,dataset_dir,is_train=True):
    self.add_class("dataset",1,"car")
    self.add_class("dataset",2,"scratch")
    images_dir=dataset_dir+'/carimg/'
    annotations_dir=dataset_dir+'/maskimg/'
    for filename in listdir(images_dir):
      image_id=filename[:-4]
      if(is_train and int(image_id)>=75):
        continue
      if not is_train and int(image_id)<75:
        continue
      img_path=images_dir+filename
      ann_path=annotations_dir+image_id+'.xml'
      self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)
  def extract_boxes(self,filename):
    tree=ElementTree.parse(filename)
    root=tree.getroot()
    boxes=list()
    names=list()
    ids=list()
    for box in root.findall('.//object'):
      xmin=int(box.find('.//bndbox/xmin').text)
      ymin=int(box.find('.//bndbox/ymin').text)
      xmax=int(box.find('.//bndbox/xmax').text)
      ymax=int(box.find('.//bndbox/ymax').text)
      name=str(box.find('name').text)
      print(xmin)
      coors=[xmin,ymin,xmax,ymax]
      boxes.append(coors)
      names.append(name)
    width=int(root.find('.//size/width').text)
    height=int(root.find('.//size/height').text)
    return boxes,width,height,names,ids
  def load_mask(self,image_id):
    info=self.image_info[image_id]
    path=info['annotation']
    boxes,w,h,names,ids=self.extract_boxes(path)
    masks=zeros([h,w,len(boxes)],dtype='uint8')
    class_ids=list()
    for i in range(len(boxes)):
      box=boxes[i]
      row_s,row_e=box[1],box[3]
      col_s,col_e=box[0],box[2]
      masks[row_s:row_e,col_s:col_e,i]=1#int(ids[i])
      class_ids.append(self.class_names.index(names[i]))
    return masks,asarray(class_ids,dtype='int32')
  def image_reference(self,image_id):
    info=self.image_info[image_id]
    return info['path']
class PredictionConfig(Config):
  NAME="car2_cfg"
  NUM_CLASSES=1+2
  GPU_COUNT=1
  IMAGES_PER_GPU=1
def plot_actual_vs_predicted(dataset,model,cfg,n_images=1):
  for i in range(n_images):
    image=dataset.load_image(i)
    mask,_=dataset.load_mask(i)
    scaled_image=mold_image(image,cfg)
    sample=expand_dims(scaled_image,0)
    yhat=model.detect(sample,verbose=0)[0]
    yhat1=model.detect(sample,verbose=0)
    pyplot.subplot(n_images,2,i*2+1)
    pyplot.imshow(image)
    pyplot.title('Actual')
    for j in range(mask.shape[2]):
      pyplot.imshow(mask[:,:,j],cmap='gray',alpha=0.3)
    pyplot.subplot(n_images,2,i*2+2)
    pyplot.imshow(image)
    pyplot.title('Predicted')
    ax = pyplot.gca()
    for box in yhat['rois']:
      y1,x1,y2,x2=box
      width,height=x2-x1,y2-y1
      rect=Rectangle((x1,y1),width,height,fill=False,color='red')
      ax.add_patch(rect)
  pyplot.show()
train_set=PersonDataset()
train_set.load_dataset('car2',is_train=True)
train_set.prepare()
print("Train: %d"%len(train_set.image_ids))
test_set=PersonDataset()
test_set.load_dataset('car2',is_train=False)
test_set.prepare()
print("Test: %d"%len(test_set.image_ids))
cfg=PredictionConfig()
model=MaskRCNN(mode='inference',model_dir='./',config=cfg)
model_path='/content/drive/My Drive/Mask_RCNN/car2_cfg20200629T1007/mask_rcnn_car2_cfg_0001.h5'
model.load_weights(model_path,by_name=True)
plot_actual_vs_predicted(train_set,model,cfg)
plot_actual_vs_predicted(test_set,model,cfg)

import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
class_names=['BG','car','scratch']
image=skimage.io.imread('/content/drive/My Drive/Mask_RCNN/tstimg.png')
results=model.detect([image],verbose=1)
r=results[0]
visualize.display_instances(image,r['rois'],r['masks'],r['class_ids'],class_names,r['scores'])