import cv2
import numpy as np
from matplotlib import pyplot as plt

class Local_Binary_pattern:
    def __init__(self):
        print("Local binary pattern object is created")
    
    def get_local_binary_pattern(self, image_file):
        img_bgr = cv2.imread(image_file)
        height, width, channel = img_bgr.shape
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        img_lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = self.__local_binary_pixel(img_gray, i, j)
        hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
        output_list = []
        output_list.append({
            "img": img_gray,
            "xlabel": "",
            "ylabel": "",
            "xtick": [],
            "ytick": [],
            "title": "Gray Image",
            "type": "gray"        
        })
        output_list.append({
            "img": img_lbp,
            "xlabel": "",
            "ylabel": "",
            "xtick": [],
            "ytick": [],
            "title": "LBP Image",
            "type": "gray"
        })    
        output_list.append({
            "img": hist_lbp,
            "xlabel": "Bins",
            "ylabel": "Number of pixels",
            "xtick": None,
            "ytick": None,
            "title": "Histogram(LBP)",
            "type": "histogram"
        })
        self.__show_output(output_list)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def __local_binary_pixel(self, img, x, y):
        center = img[x][y]
        val_ar = []
        val_ar.append(self.__get_pixel(img, center, x-1, y+1))     # top right
        val_ar.append(self.__get_pixel(img, center, x, y+1))       # right
        val_ar.append(self.__get_pixel(img, center, x+1, y+1))     # bottom right
        val_ar.append(self.__get_pixel(img, center, x+1, y))       # bottom
        val_ar.append(self.__get_pixel(img, center, x+1, y-1))     # bottom left
        val_ar.append(self.__get_pixel(img, center, x, y-1))       # left
        val_ar.append(self.__get_pixel(img, center, x-1, y-1))     # top left
        val_ar.append(self.__get_pixel(img, center, x-1, y))       # top

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val+= val_ar[i]*power_val[i]
        return val

    def __show_output(self, output_list):
        output_list_len = len(output_list)
        figure = plt.figure()
        for i in range(output_list_len):
            current_dict = output_list[i]
            current_img = current_dict["img"]
            current_xlabel = current_dict["xlabel"]
            current_ylabel = current_dict["ylabel"]
            current_xtick = current_dict["xtick"]
            current_ytick = current_dict["ytick"]
            current_title = current_dict["title"]
            current_type = current_dict["type"]
            current_plot = figure.add_subplot(1, output_list_len, i+1)
            if current_type == "gray":
                current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
                current_plot.set_title(current_title)
                current_plot.set_xticks(current_xtick)
                current_plot.set_yticks(current_ytick)
                current_plot.set_xlabel(current_xlabel)
                current_plot.set_ylabel(current_ylabel)
            elif current_type == "histogram":
                current_plot.plot(current_img, color = "black")
                current_plot.set_xlim([0,260])
                current_plot.set_title(current_title)
                current_plot.set_xlabel(current_xlabel)
                current_plot.set_ylabel(current_ylabel)            
                ytick_list = [int(i) for i in current_plot.get_yticks()]
                current_plot.set_yticklabels(ytick_list,rotation = 90)

        plt.show()


if __name__ == "__main__":
    image_file = './image_2.png'
    lbp = Local_Binary_pattern()
    lbp.get_local_binary_pattern(image_file)