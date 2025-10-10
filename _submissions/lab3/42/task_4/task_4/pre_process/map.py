import time
import yaml
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageOps

from task_4.utils.ds import Node, Tree


class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self, map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        # map_name = map_df.image[0]
        map_name = map_name + '.pgm'
        im = Image.open(map_name)
        # size = 200, 200
        # im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax]

    def __get_obstacle_map(self, map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()), (self.map_im.size[1], self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array


# print(Map('sync_classroom_map'))
# print(Map('classroom_map'))


class MapProcessor():
    def __init__(self, name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self, map_array, i, j, value, absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self, kernel, map_array, i, j, absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array, i, j, kernel[0][0], absolute)
        else:
            for k in range(i-dx, i+dx):
                for l in range(j-dy, j+dy):
                    self.__modify_map_pixel(map_array, k, l, kernel[k-i+dx][l-j+dy], absolute)

    def inflate_map(self, kernel, absolute=True):
        ## Normalization of map-obstacles
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1, j)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_up], [1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1, j)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_dw], [1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i, j-1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_lf], [1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i, j+1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_rg], [1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1, j-1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_up_lf], 
                                                                          [np.sqrt(2)]
                                                                          # [0.5]  ## [Does not work] Making diagnols have lesser weight for smooth and direct movement, configuration specific for A*
                                                                          )
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1, j+1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_up_rg], 
                                                                          [np.sqrt(2)]
                                                                          # [0.5]  ## [Does not work] Making diagnols have lesser weight for smooth and direct movement, configuration specific for A*
                                                                          )
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1, j-1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_dw_lf], 
                                                                          [np.sqrt(2)]
                                                                          # [0.5]  ## [Does not work] Making diagnols have lesser weight for smooth and direct movement, configuration specific for A*
                                                                          )
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1, j+1)]
                            self.map_graph.g['%d,%d'%(i, j)].add_children([child_dw_rg], 
                                                                          [np.sqrt(2)]
                                                                          # [0.5]  ## [Does not work] Making diagnols have lesser weight for smooth and direct movement, configuration specific for A*
                                                                          )

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self, path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    tup = tuple([i, j])
                    # print(tup)
                    path_tuple_list.append(tup)
                    if i%2 == 0 and j%2 == 0:
                        path_array[tup] = 0.25
                    # print(path_array[tup])
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.75
        return path_array
    
    def draw_nodes(self):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    tup = tuple([i, j])
                    # print(tup)
                    path_tuple_list.append(tup)
                    if i%2 == 0 and j%2 == 0:
                        path_array[tup] = 0.25
                    # print(path_array[tup])
        return path_array


# mp = MapProcessor('sync_classroom_map')
# mp = MapProcessor('classroom_map')

# kr = mp.rect_kernel(5, 1)
# #kr = mp.rect_kernel(1, 1)
# mp.inflate_map(kr, True)

# mp.get_graph_from_map()

# fig, ax = plt.subplots(dpi=100)
# plt.imshow(mp.inf_map_img_array)
# plt.colorbar()
# plt.show()



# mp.map_graph.root = "100,12"
# mp.map_graph.end = "130,180"

# # mp.map_graph.root = "100,12"
# # mp.map_graph.end = "50,112"

# # mp.map_graph.root = "10,20"
# # mp.map_graph.end = "10,185"

# # mp.map_graph.root = "30,8"
# # mp.map_graph.end = "35,45"

# start = time.time()
# as_maze = AStar(mp.map_graph)
# end = time.time()
# print('Elapsed Init Time: %.3f'%(end - start))

# start = time.time()
# as_maze.solve(mp.map_graph.g[mp.map_graph.root], mp.map_graph.g[mp.map_graph.end])
# end = time.time()
# print('Elapsed Solve Time: %.3f'%(end - start))

# path_as, dist_as = as_maze.reconstruct_path(mp.map_graph.g[mp.map_graph.root], mp.map_graph.g[mp.map_graph.end])

# path_arr_as = mp.draw_path(path_as)

# path_as

# fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=300, sharex=True, sharey=True)
# # fig, ax = plt.subplots(nrows = 1, ncols = 2, dpi=300, sharex=True, sharey=True)
# # ax[0].imshow(path_arr_bfs)
# # ax[0].set_title('Path BFS')
# # ax[1].imshow(path_arr_djk)
# # ax[1].set_title('Path Dijkstra')
# # ax[2].imshow(path_arr_as)
# # ax[2].set_title('Path A*')
# ax.imshow(path_arr_as)
# ax.set_title('Path A*')

# plt.show()

# # print(dist_bfs)
# # print(dist_djk)
# print(dist_as)