import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import gym.spaces as spaces


class ToyGridEnv(gym.Env):
    EMPTY = [255, 255, 255]
    WALL = [0, 0, 0]

    def __init__(self, seed, max_episode_length, action_repeat, obstacle_num= 100 , lidar_num=20):

        # def __init__(self, map_size=19, seed=None, obstacle_num=50, lidar_num=20):
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (-1., 1.)
        self.spec = None
        self.symbolic = True

        # Set these in ALL subclasses

        # init
        self.state = None
        self.world = None
        self.max_step = max_episode_length

        self.lidar_num = lidar_num
        self.obstacle_num = obstacle_num
        self.observation_space = 2 + self.lidar_num + 2
        self.map_size = 19

        self.action_repeat = action_repeat

        self.action_space = spaces.Box(np.float32([-1, -1]), np.float32([1, 1]), dtype=np.float32)
        self.seed(seed)
        self.recent_pose = []
        self.reset()

    def close(self):
        return None

    def set_obstacle_num(self, obstacle_num):
        self.obstacle_num = obstacle_num
        self.reset()

    def reset(self):
        self._make_world()
        self.state = np.zeros(4 + self.lidar_num)
        self.state[0:2] = self.init
        self.state[2:2 + self.lidar_num] = self._range_find()
        self.state[2 + self.lidar_num:] = self.goal
        self.t = 0
        return torch.tensor(self.state, dtype=torch.float32).unsqueeze(dim=0)

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        return [seed]

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or (self.t == self.max_step)
            if done:
                break
        observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        return observation, reward, done

    def _step(self, act):
        new_state = np.copy(self.state)
        old_state = np.copy(self.state)
        act = act / np.linalg.norm(act) if np.linalg.norm(act) > 1 else act
        new_state[0] += act[0]
        new_state[1] += act[1]

        collision = self._check_collision(new_state)
        if collision:
            pass
        else:
            self.state[0:2] = np.copy(new_state[0:2])
            self.state[2:2 + self.lidar_num] = np.copy(self._range_find())
            self.state[2 + self.lidar_num:] = self.goal

        distance = math.sqrt((self.state[0] - self.goal[0]) ** 2 + (self.state[1] - self.goal[1]) ** 2)
        old_distance = math.sqrt((old_state[0] - self.goal[0]) ** 2 + (old_state[1] - self.goal[1]) ** 2)
        is_arrived = distance <= 1
        if is_arrived:
            print("!!!!!!!!!!!!!!!!!!arrived!!!!!!!!!!!!!!!!!!!!!!")
        reward = - collision - distance + 10 * is_arrived
        done = is_arrived
        info = [distance, collision, is_arrived]
        return (self.state.copy(), reward, done, info)

    def _range_find(self):
        img = cv2.cvtColor(self.world.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imshow_offset = 0.5
        x1 = self.state[0] + imshow_offset
        y1 = self.state[1] + imshow_offset
        statesize = self.lidar_num
        range_state = np.zeros(statesize)
        theta_step = 2 * math.pi / statesize
        theta_list = [i * theta_step for i in range(statesize)]
        for st_idx, theta in enumerate(theta_list):
            x = [x1 + i * self.map_size / 100 * math.cos(theta) for i in range(100)]
            y = [y1 + i * self.map_size / 100 * math.sin(theta) for i in range(100)]
            find_range = False
            for i in range(len(x)):
                truncation_fix_x = [0.01, -0.01]
                truncation_fix_y = [0.01, -0.01]
                for dx in truncation_fix_x:
                    for dy in truncation_fix_y:
                        if img[int(y[i] + dy), int(x[i] + dx)] == 0:
                            range_state[st_idx] = math.sqrt((x[i] - x1) ** 2 + (y[i] - y1) ** 2)
                            find_range = True
                            break
                    if find_range:
                        break
                if find_range:
                    break
        return range_state

    def render(self):
        scale = 100
        grid = np.repeat(self.world, scale, axis=0)
        grid = np.repeat(grid, scale, axis=1)
        grid = cv2.cvtColor(grid.astype(np.uint8), cv2.COLOR_RGB2BGR)

        grid = cv2.circle(grid, (int((self.init[0] + 0.5) * scale), int((self.init[1] + 0.5) * scale)), 30,
                          (0, 0, 255), thickness=-1, lineType=8)
        grid = cv2.circle(grid, (int((self.goal[0] + 0.5) * scale), int((self.goal[1] + 0.5) * scale)), 30,
                          (0, 255, 255), thickness=-1, lineType=8)

        # get croped image around the agent
        x = int((self.state[0] + 0.5) * scale)
        y = int((self.state[1] + 0.5) * scale)
        if y - 5 * scale / 2 < 0:
            y_min = 0
            y_max = 5 * scale
        elif y + 5 * scale / 2 > grid.shape[0]:
            y_min = grid.shape[0] - 5 * scale
            y_max = grid.shape[0]
        else:
            y_min = y - 5 * scale / 2
            y_max = y + 5 * scale / 2
        if x - 5 * scale / 2 < 0:
            x_min = 0
            x_max = 5 * scale
        elif x + 5 * scale / 2 > grid.shape[1]:
            x_min = grid.shape[1] - 5 * scale
            x_max = grid.shape[1]
        else:
            x_min = x - 5 * scale / 2
            x_max = x + 5 * scale / 2
        grid = grid[int(y_min):int(y_max), int(x_min):int(x_max), :]
        return grid

    @property
    def observation_size(self):
        return self.observation_space

    @property
    def action_size(self):
        return self.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self.action_space.sample())

    def _check_collision(self, pos):
        img = cv2.cvtColor(self.world.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imshow_offset = 0.5
        x1 = self.state[0] + imshow_offset
        x2 = pos[0] + imshow_offset
        y1 = self.state[1] + imshow_offset
        y2 = pos[1] + imshow_offset

        if (int(x1) == int(x2)) or (int(y1) == int(y2)):
            if (img[int(y1), int(x1)] != 0) and (img[int(y2), int(x2)] != 0):
                # print("no collision")
                return False
            else:
                # print("collision")
                return True

        resolution = 100
        x = [(x1 * (resolution - i) + i * x2) / resolution for i in range(resolution + 1)]
        y = [(y1 * (resolution - i) + i * y2) / resolution for i in range(resolution + 1)]
        truncation_fix_x = [0.01, -0.01]
        truncation_fix_y = [0.01, -0.01]

        # print("collision",x,y)
        if img[int(y[-1]), int(x[-1])] == 0:
            return True  # collision

        for i in range(len(x)):
            # print(int(x[i]),int(y[i]))
            try:
                for dx in truncation_fix_x:
                    for dy in truncation_fix_y:
                        if img[int(y[i] + dy), int(x[i] + dx)] == 0:
                            return True  # collision
            except:
                print("index error detected!")
                continue
        return False  # no-collision

    def _make_world(self):
        while True:
            # goal init making
            self.goal = np.array([np.random.uniform(2, self.map_size - 2), np.random.uniform(2, self.map_size - 2)])
            while True:
                self.init = np.array(
                    [np.random.uniform(2, self.map_size - 2), np.random.uniform(2, self.map_size - 2)])
                if math.sqrt(
                        (self.init[0] - self.goal[0]) ** 2 + (self.init[1] - self.goal[1]) ** 2) > self.map_size / 3:
                    break

            # self.state = np.zeros(2 + self.lidar_num)
            # self.state[0:2] = self.init

            world = self.EMPTY * np.ones([self.map_size, self.map_size, 3], dtype=int)

            ## side wall
            world[0, :] = self.WALL
            world[-1, :] = self.WALL
            world[:, 0] = self.WALL
            world[:, -1] = self.WALL

            obj_spawn = 0
            while obj_spawn != self.obstacle_num:
                wall_inx = np.random.randint(1, self.map_size - 1, 2)
                if not np.array_equal(world[wall_inx[0], wall_inx[1]], self.WALL):

                    # Note : wall_inx[1] = x coord, wall_inx[0] = y coord
                    # world[y,x] = world[wall_inx[0], wall_inx[1]]

                    if math.sqrt(((wall_inx[1] - self.goal[0]) ** 2) + ((wall_inx[0] - self.goal[1]) ** 2)) > 2 \
                            and math.sqrt(
                        ((wall_inx[1] - self.init[0]) ** 2) + ((wall_inx[0] - self.init[1]) ** 2)) > 2:
                        world[wall_inx[0], wall_inx[1]] = self.WALL
                        obj_spawn += 1

            self.world = world

            # check the feasibility
            demonstrator = ToyGridDemonstrator(self)
            for i in range(5):
                path = demonstrator.demonstrate()
                if path is not None:
                    break
            if path is not None:
                break

    def render_path(self, path_list=None, label_list=None,
                    save_path=None, show_figure=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        scale = 100
        grid = np.repeat(self.world, scale, axis=0)
        grid = np.repeat(grid, scale, axis=1)
        grid = cv2.cvtColor(grid.astype(np.uint8), cv2.COLOR_RGB2BGR)

        if path_list is not None:
            ## set color
            color_list = []
            plt_label_list = []
            ax_labels_list = []
            ax_color_list = []
            ax_int_list = []
            if label_list is not None:
                for label in label_list:
                    color = ['C%d' % i for i in label]
                    color_list.append(color)
                    plt_label = ['node %d' % i for i in label]
                    plt_label_list.append(plt_label)
                    ax_labels_list += plt_label
                    ax_color_list += color
                    ax_int_list += ['%04d' % i for i in label]
                ax_labels_list = list(dict.fromkeys(ax_labels_list))
                ax_color_list = list(dict.fromkeys(ax_color_list))
                ax_int_list = list(dict.fromkeys(ax_int_list))
            else:
                color_list = ['C0'] * len(path_list)
                plt_label_list = [''] * len(path_list)
                ax_labels_list = ['']
                ax_color_list = ['C0']

            ## plot path
            for path, color, plt_label in zip(path_list, color_list, plt_label_list):
                path_array = np.array(path)
                # ax.plot(path_array[:,0], path_array[:,1], alpha = 0.3, color = color)
                plot_offset = 0
                ax.plot(path_array[:, 0] + plot_offset, path_array[:, 1] + plot_offset, alpha=0.2, color='r', zorder=1)
                ax.scatter(path_array[:, 0] + plot_offset,
                           path_array[:, 1] + plot_offset,
                           s=20,
                           alpha=0.5,
                           color=color,
                           zorder=2)
                # if plot_lidar:
                #     grid = self._render_lidar(grid, path)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 2

            grid = cv2.circle(grid, (int((self.init[0] + 0.5) * scale), int((self.init[1] + 0.5) * scale)), 30,
                              (0, 0, 255), thickness=-1, lineType=8)
            grid = cv2.circle(grid, (int((self.goal[0] + 0.5) * scale), int((self.goal[1] + 0.5) * scale)), 30,
                              (0, 255, 255), thickness=-1, lineType=8)
            cv2.putText(grid, "initial",
                        (int((self.init[0] + 0.5 - 0.8) * scale), int((self.init[1] + 0.5 - 0.3) * scale)),
                        font, font_size, (0, 0, 0), 10)
            cv2.putText(grid, "goal",
                        (int((self.goal[0] + 0.5 - 0.8) * scale), int((self.goal[1] + 0.5 - 0.3) * scale)),
                        font, font_size, (0, 0, 0), 10)
            # (left, right, bottom, top)
            ax.imshow(grid, extent=(-0.5, self.map_size - 0.5, self.map_size - 0.5, -0.5))

            for ax_int in sorted(ax_int_list):
                idx = ax_int_list.index(ax_int)
                ax_color = ax_color_list[idx]
                ax_label = ax_labels_list[idx]

                ax.scatter([],
                           [],
                           color=ax_color,
                           label=ax_label,
                           alpha=1)

        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        # ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

        if save_path is not None:
            fig.savefig(save_path)
        if show_figure:
            plt.show()
        return fig


class ToyGridDemonstrator():
    def __init__(self, env: ToyGridEnv):
        self.env = env

    def demonstrate(self):
        init_state = np.copy(self.env.init)
        final_state = np.copy(self.env.goal)
        step_size = 1
        rrt = RRT(self.env.world, init_state, final_state, step_size)
        path = rrt.solve()
        return path


class Nodes:
    """Class to store the RRT graph"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []


class RRT():
    def __init__(self, img, start, end, step_size, max_search=1000):
        self.node_list = [0]
        self.img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.start = start
        self.end = end
        self.step_size = step_size
        self.max_search = max_search

    def solve(self):
        '''
        img: gray img
        img_gray: colored img
        '''
        img = np.copy(self.img)
        img_gray = self.img_gray
        node_list = self.node_list
        start = self.start
        end = self.end
        draw = False

        h, l = img_gray.shape  # dim of the loaded image
        # print(img.shape) # (384, 683)
        # print(h,l)

        # insert the starting point in the node class
        # node_list = [0] # list to store all the node points
        node_list[0] = Nodes(start[0], start[1])
        node_list[0].parent_x.append(start[0])
        node_list[0].parent_y.append(start[1])

        i = 1;
        count = 0
        while count < self.max_search:
            nx, ny = self._rnd_point(h, l)
            # print("Random points:",nx,ny)

            nearest_ind = self._nearest_node(nx, ny)
            nearest_x = node_list[nearest_ind].x
            nearest_y = node_list[nearest_ind].y
            # print("Nearest node coordinates:",nearest_x,nearest_y)

            # check direct connection
            tx, ty, directCon, nodeCon = self._check_collision(nx, ny, nearest_x, nearest_y)
            if tx is None:  # out of map point
                continue

            # find path
            if directCon and nodeCon:
                print("Node can connect directly with end")
                node_list.append(i)
                node_list[i] = Nodes(tx, ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)

                path = []
                for j in range(len(node_list[i].parent_x) - 1):
                    path.append(np.asarray([node_list[i].parent_x[j], node_list[i].parent_y[j]]))
                path.append(np.asarray([end[0], end[1]]))
                return path

            # find connected node
            elif nodeCon:
                # print("Nodes connected")
                node_list.append(i)
                node_list[i] = Nodes(tx, ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                # print(i)
                # print(node_list[nearest_ind].parent_y)
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)
                i = i + 1
                # display
                # cv2.circle(img, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                # cv2.line(img, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                # cv2.imwrite("media/"+str(i)+".jpg",img)
                # cv2.imshow("sdc",img)
                # cv2.waitKey(1)
                count += 1
                continue

            # sampled point is not reachable to any node
            else:
                # print("No direct con. and no node con. :( Generating new rnd numbers")
                count += 1
                continue

        print("RRT can not solve path finding")
        return None

    # check collision
    def _collision(self, x1, y1, x2, y2):
        img = self.img_gray
        imshow_offset = 0.5
        x1 = x1 + imshow_offset
        x2 = x2 + imshow_offset
        y1 = y1 + imshow_offset
        y2 = y2 + imshow_offset

        color = []
        x = [(x1 * (100 - i) + i * x2) / 100 for i in range(101)]
        y = [(y1 * (100 - i) + i * y2) / 100 for i in range(101)]
        truncation_fix_x = [0.01, -0.01]
        truncation_fix_y = [0.01, -0.01]

        # print("collision",x,y)
        for i in range(len(x)):
            # print(int(x[i]),int(y[i]))
            try:
                for dx in truncation_fix_x:
                    for dy in truncation_fix_y:
                        if img[int(y[i] + dy), int(x[i] + dx)] == 0:
                            return True  # collision
            except:
                print("index error detected!")
                continue
        return False  # no-collision

    # check the  collision with obstacle and trim
    def _check_collision(self, x1, y1, x2, y2):  # x1,y1 is new sampled point
        img_gray = self.img_gray
        step_size = self.step_size

        _, theta = self._dist_and_angle(x2, y2, x1, y1)
        x = x2 + step_size * np.cos(theta)
        y = y2 + step_size * np.sin(theta)

        # TODO: trim the branch if its going out of image area
        # print("Image shape",img.shape)
        hy, hx = img_gray.shape
        if y < 0 or y > hy or x < 0 or x > hx:
            # print("Point out of image bound")
            # print((x,y))
            y = None
            x = None
            directCon = False
            nodeCon = False
        else:
            # check direct connection
            dist_from_end = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
            if self._collision(x, y, self.end[0], self.end[1]) or (dist_from_end > self.step_size):
                directCon = False
            else:
                directCon = True

            # check connection between two nodes
            if self._collision(x, y, x2, y2):
                nodeCon = False
            else:
                nodeCon = True

        return (x, y, directCon, nodeCon)

    # return dist and angle b/w new point and nearest node
    def _dist_and_angle(self, x1, y1, x2, y2):
        dist = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        angle = math.atan2(y2 - y1, x2 - x1)
        return (dist, angle)

    # return the neaerst node index
    def _nearest_node(self, x, y):
        node_list = self.node_list
        temp_dist = []
        for i in range(len(node_list)):
            dist, _ = self._dist_and_angle(x, y, node_list[i].x, node_list[i].y)
            temp_dist.append(dist)
        return temp_dist.index(min(temp_dist))

    # generate a random point in the image space
    def _rnd_point(self, h, l):
        new_y = np.random.randint(0, h)
        new_x = np.random.randint(0, l)
        return (new_x, new_y)
