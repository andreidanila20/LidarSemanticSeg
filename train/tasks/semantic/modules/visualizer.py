# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
from .laserscan import LaserScan, SemLaserScan
import numpy as np
import yaml
import pptk


WINDOW_NAME = "Vizualizer"


class Visualizer():
    def __init__(self, HEIGHT=64, WIDTH=1024, scan_files=None, label_files=None, prediction_files=None, med_filter=False):
        super().__init__()
        self.DATA_CFG = yaml.safe_load(open("C:\\Users\\Dedu\\PycharmProjects\\RangeNet++\\semantic-kitti.yaml", 'r'))

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.play_all = False

        self.med_filter_flag = med_filter

        self.point_clouds = []
        self.labels = []
        self.predictions = []
        self.iteration_idx = 0

        if (scan_files == None):
            print("You are required to introduce a data path")
        else:
            self.point_clouds = scan_files
            self.labels = label_files

        print(self.point_clouds[0])

        if not (prediction_files == None):
            self.predictions = prediction_files

        self.scan = LaserScan(project=True, H=HEIGHT, W=WIDTH, fov_up=3, fov_down=-25)
        self.semantic_scan = SemLaserScan(sem_color_dict=self.DATA_CFG['color_map'], project=True, H=HEIGHT, W=WIDTH,
                                          fov_up=3,
                                          fov_down=-25)

    def computeImage(self):

        self.scan.open_scan(self.point_clouds[self.iteration_idx])

        self.semantic_scan.open_scan(self.point_clouds[self.iteration_idx])
        self.semantic_scan.open_label(self.labels[self.iteration_idx])
        self.semantic_scan.colorize()

        proj_depth = self.scan.proj_range
        max_depth = max(proj_depth[proj_depth > 0])
        proj_depth = proj_depth / max_depth
        image_grayscale = cv2.cvtColor(proj_depth, cv2.COLOR_GRAY2RGB)

        image_color = self.semantic_scan.proj_sem_color


        if self.predictions.__len__() > 0:
            self.semantic_scan.open_scan(self.point_clouds[self.iteration_idx])
            self.semantic_scan.open_label(self.predictions[self.iteration_idx])
            self.semantic_scan.colorize()

            predicted_image = self.semantic_scan.proj_sem_color

            all_images = np.concatenate((image_grayscale, image_color), axis=0)
            all_images = np.concatenate((all_images, predicted_image), axis=0)
            if self.med_filter_flag:
                filtered = self.median_filter(predicted_image)
                all_images = np.concatenate((all_images, filtered), axis=0)

        else:
            all_images = np.concatenate((image_grayscale, image_color), axis=0)

        return all_images


    def median_filter(self, img):

        image = img

        med = 5

        for i in range(1, self.HEIGHT-1):
            for j in range(1, self.WIDTH-1):
                filtered = []
                if(image[i][j].sum() == 0.):
                    for k in range(0, 3):
                        for l in range(0, 3):
                            filtered.append(image[i+k-1][j+l-1])


                    x = np.sort(filtered, axis=0)
                    image[i][j] = x[med]

        return image

    def view_all(self):
        for i in range(self.iteration_idx, self.point_clouds.__len__()):
            if self.play_all == False:
                break
            image = self.computeImage()
            cv2.imshow(WINDOW_NAME, image)
            cv2.waitKey(1)
            self.iteration_idx = self.iteration_idx + 1
        if(self.iteration_idx==self.point_clouds.__len__()):
            cv2.destroyAllWindows()

    def mouse_click(self, event, x, y,
                    flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # font for left click event
            third_width = int(self.WIDTH * 0.3)
            print(x)
            if x < third_width:
                if self.iteration_idx > 0: self.iteration_idx = self.iteration_idx - 1
            elif x > third_width * 2:
                if self.iteration_idx < self.point_clouds.__len__(): self.iteration_idx = self.iteration_idx + 1
            else:
                self.play_all = not (self.play_all)

            if self.play_all == False:
                image = self.computeImage()
                cv2.imshow(WINDOW_NAME, image)
                cv2.waitKey(1)
            else:
                self.view_all()

        # to check if right mouse
        # button was clicked
        if event == cv2.EVENT_RBUTTONDOWN:
            print(x, y)

    def visualize_projections(self):
        scans = (self.scan, self.semantic_scan)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_click, param=scans)

    def get_pcd_viewer(self, index):
        rgb = []
        pc_data = np.fromfile(self.point_clouds[index], '<f4')
        pc_data = np.reshape(pc_data, (-1, 4))

        pc_color = np.fromfile(self.labels[index], dtype=np.int32)
        pc_color = np.reshape(pc_color, -1)
        pc_color = pc_color & 0xFFFF

        for i in range(0, len(pc_color)):
            rgb.append(self.DATA_CFG['color_map'][pc_color[i]])

        rgb = np.asarray(rgb) / 255.
        print(len(rgb), " ", len(pc_data))

        xyz_data = pc_data[:, :3]
        viewer = pptk.viewer(xyz_data)
        viewer.attributes(rgb)

        return viewer

    def vizualize_pcd(self):
        viewer = None
        for i in range(0, self.point_clouds.__len__()):
            if viewer == None:
                viewer = self.get_pcd_viewer(i)
            else:
                key = input("Press Enter to continue or type exit to close the viewer...")
                if (key == 'exit'):
                    viewer.close()
                    break
                viewer.close()
                viewer = self.get_pcd_viewer(i)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
