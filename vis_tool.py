#!/usr/bin/env python
""" Visualization tool for Tensorflow models.
 
Usage:
  run as:
   python visualization_real_time.py --pb=<THE_PB_FILE_PATH> --gpu=False

  example:
   python visualization_real_time.py --pb='classifier.pb' --gpu=False
"""
from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
import math
import re
import libavg
from libavg import avg, app, player, widget


class VisualizationTool(app.MainDiv):
  # An OptionParser instance is passed to this function, allowing the MainDiv to
  # add command line arguments
  def onArgvParserCreated(self, parser):
      parser.add_option('--pb', '-p', default='/', dest='pb_file',
              help='.pb file')
      parser.add_option('--gpu', '-g', default=False, dest='gpu',
              help='Use GPU')

  def onArgvParsed(self, options, args, parser):
    """ This method is called when the command line options are being parsed.
        options, args are the result of OptionParser.parse_args().
    """
    self.argvoptions = options

  def onStartup(self):
    """ Called before libavg has been setup, just after the App().run() call. 
        The window has not been created at this time.
    """
    player.setWindowTitle('ConvNets Visualization')

  def onInit(self):
    """ This is called as soon as the player is started by the App object.
        Initialize everything here.
    """
    self.act_maps_nodes = list()
    self.sep_tiles = 5
    self.act_maps_imgs = list() # contains all the activation maps of the current layer
    self.selected_act_map_idx = 0
    self.num_maps = 0
    self.w_names = list()                 # weights tensors names
    self.w_names_short = list()           # layer weights names
    self.act_maps_names_tensors = list()  # activation maps tensors names
    self.act_maps_names_short = list()    # activation maps names
    self.current_shape = []               # shape of current layer or weight

    #Location and name of trained model
    self.trained_model_dir = self.argvoptions.pb_file

    # Load the model (graph and weights) on cpu or gpu 
    self.graph = self.load_frozen_graph(self.trained_model_dir, on_gpu=self.argvoptions.gpu)

    # Start running operations on the Graph
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
                      allow_soft_placement=True))

    # get the list of operations from the graph
    ops = self.graph.get_operations()
    print(len(ops))
    ops_lst = list()
    for i in range(len(ops)):
      print(ops[i].name)
      ops_lst.append(ops[i].name)

    # Get the input placeholder
    self.net_input_placeholder = self.graph.get_tensor_by_name(ops[0].name + ':0')
    self.model_image_size = self.net_input_placeholder.get_shape().as_list()[1]

    # get the list of weights names tensors
    for i in range(len(ops_lst)):
      if ops_lst[i].endswith('weights'):
        self.w_names.append(ops_lst[i] + ':0') 
        self.w_names_short.append((ops_lst[i])[:-7])
        
    # get the list of activation maps tensors 
    # activation map is the last operation of each trainable layer (layer with weights)
    for i in range(len(self.w_names_short)):
      l_name = (self.w_names_short[i])   # get the name of the trainable layer (name before "/weights")

      r = re.compile(l_name)
      tensors = filter(r.match, ops_lst) # search for all the operations of this layer
      self.act_maps_names_tensors.append(tensors[-1] + ':0')  # get the last operation
      self.act_maps_names_short.append(tensors[-1])
    
    # Get the activation maps of the first layer
    self.current_layer_idx = 0
    frame = np.zeros((self.model_image_size,self.model_image_size,3))
    frame = np.expand_dims(frame, axis=0)
    units = self.get_activation_maps(frame, self.current_layer_idx)
    self.num_maps = units.shape[3]
    self.current_shape = units.shape

    # set a white background for the window
    avg.RectNode(parent=self, size=self.size, fillopacity=0.5)

    # Create the image boxes
    self.build_window(self.num_maps)

    # add a keyboard listener
    player.subscribe(player.KEY_DOWN, self.onKey)
    
    self.cap = cv2.VideoCapture(0)

  def onExit(self):
        print('Exiting Visualization..')

  def onFrame(self):
    """ 
      This method is called continuously
      It reads a new frame from the camera, obtain the activation maps
      and update the tiles.
    """

    # read frame from camera
    ret, frame = self.cap.read()

    # prepare input frame for network
    frame = cv2.resize(frame, (self.model_image_size,self.model_image_size))
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  # convert to RGB
    frameNorm = (frameRGB.astype('float')) - 128      # normalize to (-128,127) 
    frameInput = np.expand_dims(frameNorm, axis=0) 

    units = self.get_activation_maps(frameInput, self.current_layer_idx)

    self.update_tiles(frameRGB, units)


  def get_activation_maps(self, image_input, layer_idx):
    """ Extract the activation maps for a particular layer
    
    Args:
        image_input: input image to the network of shape (1,model_image_size, model_image_size, 3)
        
    Returns: 
        units: numpy array containing the activation maps
    """

    # get activation maps of thelayer_idx layer
    out_map = self.graph.get_tensor_by_name(self.act_maps_names_tensors[layer_idx])
    units = self.sess.run(out_map,feed_dict={self.net_input_placeholder:image_input})

    return units


  def build_window(self,num_images):
    """ Create all the window components (nodes) in three columns as follows

        Leftcol    Center     right_col
       __________________________
      |      |              |    |
      |      |              |    |
      |      |              |    |
      |      |              |    |
      |______|______________|____|
    
    Args:
        num_images: number of images to display.
        
    Returns: 
        None
    """

    self.num_maps = num_images

    res = self.settings.getPoint2D('app_resolution')
    self.win_size_x = res[0]
    self.win_size_y = res[1]

    # save window space for the left column (big images)
    self.left_col_size = 200
    self.frame_size = self.left_col_size - self.sep_tiles # video frame size

    # space size for right column (layers names)
    right_col_size = 200

    # tiles column size
    tile_win_size = self.win_size_x - self.left_col_size - right_col_size - 10

    # ADD THE NODES OF THE RIGHT COLUMN
    posx = self.left_col_size + tile_win_size + 5
    posy = 10
    self.buttons_label = avg.WordsNode(text='Activation Maps', pos=(posx,posy), parent=player.getRootNode())
    posy += 20
    self.display_buttons_layers(posx, posy, right_col_size, 15)

    # ADD THE NODES OF THE LEFT COLUMN
    posx = 10
    posy = 5
    self.layer_label = avg.WordsNode(text='Layer: ', pos=(posx,posy), parent=player.getRootNode())
    posy += 17
    self.layer_name_node = avg.WordsNode(text=self.act_maps_names_short[self.current_layer_idx], pos=(posx,posy), parent=player.getRootNode())
    posy += 17
    self.layer_shape = avg.WordsNode(text=str(self.current_shape), pos=(posx,posy), parent=player.getRootNode())
    
  
    # current video frame label
    posy += 30 
    self.frame_node_label = avg.WordsNode(text='Current frame', pos=(posx,posy), parent=player.getRootNode())
    # current video frame image 
    posx= 1
    posy += 20
    self.frame_node = avg.ImageNode(href="", pos=(posx,posy),parent=player.getRootNode())

    # current selected tile label
    posy += self.frame_size + 20
    string = "Tensor %d of %d" % (self.selected_act_map_idx + 1,num_images)
    self.selected_act_map_text_node = avg.WordsNode(text=string, pos=(posx,posy), parent=player.getRootNode())
    # current selected tile image
    posx = 1
    posy += 20
    self.selected_act_map_node = avg.ImageNode(href="", pos=(posx,posy),parent=player.getRootNode())

    # PUT THE NODES OF THE CENTER COLUMN
    # calculate the tile size in order to fit all the images
    # in the given resolution, leaving space for left column
    self.cols = math.ceil(math.sqrt(num_images))
    self.tile_size = tile_win_size/self.cols
    self.act_maps_size = self.tile_size - self.sep_tiles

    posx = self.left_col_size
    posy = 10
    # activation maps
    self.act_maps_nodes = list()
    for i in range(0,num_images):
      pos = avg.Point2D(posx, posy) + ((i%self.cols)*self.tile_size, (i//self.cols)*self.tile_size)
      imgNode = avg.ImageNode(href="", pos=pos, parent=player.getRootNode())
      
      # suscribe to mouse-down event
      imgNode.subscribe(avg.Node.CURSOR_DOWN, self.on_activation_down)
      
      self.act_maps_nodes.append(imgNode)

    #node = widget.TextButton(pos=(10,900), size=(100,20), text='Delete', parent=self)
    #node.subscribe(widget.Button.CLICKED, self.on_button_down2)

  def delete_nodes(self):
    """ Delete all the nodes from the central column window. 
        This is called each time the user selects a different layer
    
    Args:
        None
        
    Returns: 
        None
    """
    self.layer_label.unlink(True)
    self.layer_name_node.unlink(True)
    self.layer_shape.unlink(True)
    self.frame_node_label.unlink(True)
    self.frame_node.unlink(True)
    self.selected_act_map_text_node.unlink(True)
    self.selected_act_map_node.unlink(True)

    for i in range(len(self.act_maps_nodes)):
      self.act_maps_nodes[i].unsubscribe(avg.Node.CURSOR_DOWN, self.on_activation_down)
      self.act_maps_nodes[i].unlink(True)
    del self.act_maps_nodes[:]

    self.buttons_label.unlink(True)
    for i in range(len(self.buttons_nodes)):
      self.buttons_nodes[i].unlink(True)
    del self.buttons_nodes[:]


  def display_buttons_layers(self,xpos,ypos,label_width,label_height):
    """ Create the buttons of the right column (layer names) inside a ScrollArea node
    
    Args:
        xpos: absolute x position on the parent window
        ypos: absolute y position on the parent window
        label_width:  button width
        label_height: button height
        
    Returns: 
        None
    """
    self.buttons_nodes = list()

    area = avg.DivNode(width=label_width+100, height=len(self.act_maps_names_short)*(label_height+2)+20)
    posx = 10
    posy = 10
    for i in range(0,len(self.act_maps_names_short)):
      node = widget.TextButton(pos=(posx,posy), size=(label_width,label_height), text=self.act_maps_names_short[i], parent=area)
      node.subscribe(widget.Button.CLICKED, lambda i=i: self.on_button_down(i))
      posy = posy + label_height + 2

      self.buttons_nodes.append(node)

    scrollArea = widget.ScrollArea(contentNode=area, parent=self, pos=(xpos,ypos), size=(label_width,self.win_size_y-50))

  def on_activation_down(self, event):
    """ Mouse callback on the activation maps. 
        This is called each time the user performs a mouse clic on an activation map.
    
    Args:
        event: mouse event
        
    Returns: 
        None
    """
    # identify tile pressed by (x,y) coordinates
    x_coord = event.x
    y_coord = event.y

    x = x_coord - self.left_col_size
    y = y_coord - 10
    xi = int(x / self.tile_size)
    yi = int(y / self.tile_size)
    self.selected_act_map_idx = int(yi * self.cols + xi)

  def on_button_down(self, i):
    """ Mouse callback on the activation maps. 
        This is called each time the user performs a mouse clic on an activation map.
    
    Args:
        event: mouse event
        
    Returns: 
        None
    """
    self.current_layer_idx = i
    self.delete_nodes()

    self.selected_act_map_idx = 0

    # Get the activation maps of the selected layer
    frame = np.zeros((self.model_image_size,self.model_image_size,3))
    frame = np.expand_dims(frame, axis=0)
    units = self.get_activation_maps(frame, self.current_layer_idx)
    self.num_maps = units.shape[3]
    self.current_shape = units.shape

    # Create the multiple images
    self.build_window(self.num_maps)
  
  def on_button_down2(self):
    self.delete_nodes()

  def onKey(self, event):
    # if curspr up, move one row up
    if event.scancode == 82:   
      self.selected_act_map_idx = int(self.selected_act_map_idx - self.cols)
      if self.selected_act_map_idx < 0:
        self.selected_act_map_idx = 0
    # if cursor down, move one row down
    elif event.scancode == 81:   
      self.selected_act_map_idx = int(self.selected_act_map_idx + self.cols)
      if self.selected_act_map_idx > self.num_maps - 1:
        self.selected_act_map_idx = self.num_maps - 1
    # if cursor left, move one column left
    elif event.scancode == 80:   
      self.selected_act_map_idx = int(self.selected_act_map_idx - 1)
      if self.selected_act_map_idx < 0:
        self.selected_act_map_idx = 0
    # if cursor right, move one column right
    elif event.scancode == 79:   
      self.selected_act_map_idx = int(self.selected_act_map_idx + 1)
      if self.selected_act_map_idx > self.num_maps - 1:
        self.selected_act_map_idx = self.num_maps - 1

  def update_tiles(self, frame, units):
    # update layer name 
    self.layer_name_node.text = self.act_maps_names_short[self.current_layer_idx]
    self.layer_shape.text = str(self.current_shape)

    # resize input frame and convert to rgba
    frame_size = int(self.frame_size)
    frame_img = cv2.resize(frame,(frame_size,frame_size))
    frame_img = cv2.cvtColor(frame_img,cv2.COLOR_RGB2RGBA)
    frame_data = frame_img.tobytes() # extract data as binary string

    # update the img frame
    bitmap = avg.Bitmap((self.frame_size,self.frame_size),avg.R8G8B8A8, "")
    bitmap.setPixels(frame_data)
    self.frame_node.setBitmap(bitmap)

    # update activation units tiles
    del self.act_maps_imgs[:]
    tiles_size = int(self.act_maps_size)
    for i in range(len(self.act_maps_nodes)):
      # get the activation unit 
      act_unit = units[0,:,:,i]

      act_unit = (act_unit/(np.amax(act_unit)))*255.0
      act_unit = act_unit.astype(int)
      
      #act_unit = cv2.applyColorMap(act_unit, cv2.COLORMAP_HOT)
      
      # resize tiles and convert to bgra
      tile_img = cv2.resize(act_unit, (tiles_size,tiles_size),interpolation=cv2.INTER_NEAREST)
      tile_img = cv2.convertScaleAbs(tile_img)
      tile_img = cv2.cvtColor(tile_img,cv2.COLOR_GRAY2RGB)
      self.act_maps_imgs.append(tile_img)

      tile_img = cv2.cvtColor(tile_img,cv2.COLOR_RGB2RGBA)
      tile_data = tile_img.tobytes()

      # create bitmap
      bitmap = avg.Bitmap((tiles_size,tiles_size),avg.R8G8B8A8, "")
      bitmap.setPixels(tile_data)

      self.act_maps_nodes[i].setBitmap(bitmap)
    
    # put border on selected tile
    img = self.act_maps_imgs[self.selected_act_map_idx]
    img_border = cv2.resize(img, (tiles_size-6,tiles_size-6))
    img_border = cv2.copyMakeBorder(img_border, 3, 3, 3, 3, borderType = 0, value = (255,255,0))
    img_border = cv2.cvtColor(img_border,cv2.COLOR_RGB2RGBA)
    tile_data = img_border.tobytes()
    bitmap = avg.Bitmap((tiles_size,tiles_size),avg.R8G8B8A8, "")
    bitmap.setPixels(tile_data)
    self.act_maps_nodes[self.selected_act_map_idx].setBitmap(bitmap)

    # update big selected tile
    string = "Tensor %d of %d" % (self.selected_act_map_idx + 1,self.num_maps)
    self.selected_act_map_text_node.text = string

    frame_size = int(self.frame_size)
    img = cv2.resize(img, (frame_size,frame_size))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    img_data = img.tobytes()
    bitmap = avg.Bitmap((frame_size,frame_size),avg.R8G8B8A8, "")
    bitmap.setPixels(img_data)

    # update the image
    self.selected_act_map_node.setBitmap(bitmap)

  def load_frozen_graph(self,frozen_graph_filename, on_gpu):
    """ Load a freezed model (.pb) into the current graph.
    
    Args:
        frozen_graph_filename: path to the freezed mode.
    Returns: 
        the loaded graph
    """
    # First we need to load the protobuf file from the disk and parse it to retrieve the 
    # Unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    if on_gpu == 'True':
      device = '/gpu:0'
    else:
      device = '/cpu:0'
      
    with tf.Graph().as_default() as graph, tf.device(device):
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name=None, 
            op_dict=None, 
            producer_op_list=None
        )
    return graph 
    


if __name__ == '__main__':
    # App options (such as app_resolution) can be changed as parameters of App().run()
    app.App().run(VisualizationTool(), app_resolution='1400x1000')


