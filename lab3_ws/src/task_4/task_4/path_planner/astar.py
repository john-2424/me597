import heapq
import itertools
import numpy as np


class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self._counter = itertools.count()  # tie-breaker to keep heap stable
        self.open_hq = []
        self.closed_set = set()
        self.dist = {name:np.inf for name, node in in_tree.g.items()}
        self.h = {name:0 for name, node in in_tree.g.items()}

        ## From every node to the end node distance in terms of coordinates or pixels on the image
        for name, node in self.in_tree.g.items():
            # start = tuple(map(int, name.split(',')))
            # end = tuple(map(int, self.in_tree.end.split(',')))
            start = node.pose
            end = self.in_tree.g[self.in_tree.end].pose
            ## Heuristic in terms of Euclidean Distance
            # self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            ## Heuristic in terms of Octile Distance
            ## Use as many diagonals as possible, min(dx,dy) diagonals, then finish the leftover |dx-dy| steps straight
            d = 1
            dr2 = np.sqrt(2)
            dx = abs(end[0]-start[0])
            dy = abs(end[1]-start[1])
            self.h[name] = d*(dx + dy) + (dr2-2*d)*(min(dx, dy))
            # self.h[name] = d*max(dx, dy) + (dr2-d)*(min(dx, dy))
            # self.h[name] = dr2*min(dx, dy) + d*(abs(dx-dy))
            ## ToDo: Heuristic to factor other costs apart from distance such as,
            ## time (bot's limitation in terms of manueverability and mobility), 
            ## space (proximity of bot to obstacles/safety distance from obstacles including the bot physical features like width/diameter), and
            ## energy (fuel - gas/e-)
            # self.h[name] = None
        
        self.via = {name:0 for name, node in in_tree.g.items()}

        # for __, node in in_tree.g.items():
        #     self.q.push(node)
    
    def __init_hq(self, sn):
        heapq.heappush(self.open_hq, (self.__get_f_score(sn), next(self._counter), sn.name))
        ## assigning the dist for the start node to zero to start the solve from it
        self.dist[sn.name] = 0
    
    def __get_f_score(self, node):
        idx = node.name
        ## upto the node distace + from the node to the end node heuristic
        return self.dist[idx] + self.h[idx]

    ## From SN to all Leaf Nodes with considering weights+heuristic and distance
    def solve(self, sn, en):
        ## Initialize Open Heap Queue with Start Node and update Start Node distance value with 0
        self.__init_hq(sn)

        while self.open_hq:
            _, _, pn_name = heapq.heappop(self.open_hq)  ## pn is parent node
            
            if pn_name in self.closed_set:
                continue
            self.closed_set.add(pn_name)

            ## to not traverse after the end node point 
            if pn_name == en.name:
                break
            
            pn = self.in_tree.g[pn_name]
            ## from the node with shortest distance to its children
            for i in range(len(pn.children)):
                cn = pn.children[i]
                cw = pn.weight[i]
                new_dist = self.dist[pn.name] + cw
                if new_dist < self.dist[cn.name]:
                    self.dist[cn.name] = new_dist
                    self.via[cn.name] = pn.name
                    heapq.heappush(self.open_hq, (self.__get_f_score(cn), next(self._counter), cn.name))

    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path, dist