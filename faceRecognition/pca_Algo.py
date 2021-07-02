import numpy as np 
import cv2
import scipy.linalg as s_linalg


class Pca_Algo:
    def __init__(self, images, img_labels, img_targets, no_of_elements,quality_percent):
        self.images = np.asarray(images)
        self.img_labels=img_labels
        self.img_targets=img_targets
        self.no_of_elements=no_of_elements
        #self.img_width=img_width
        #self.img_height=img_height
        self.quality_percent=quality_percent

        mean = np.mean(self.images, 1)
        self.mean_face = np.asmatrix(mean).T
        self.images =  self.images - self.mean_face

    def give_p_value(self, eig_vals):
        sum = np.sum(eig_vals)
        sum_85 = self.quality_percent * sum / 100
        temp = 0
        p = 0
        while temp < sum_85:
            temp += eig_vals[p]
            p += 1
        return p

    def reduce_dim(self):
        p, d, q = np.linalg.svd(self.images, full_matrices=True)
        p_matrix = np.matrix(p)
        d_diag = np.diag(d)
        q_matrix = np.matrix(q)
        p = self.give_p_value(d)
        self.new_bases = p_matrix[:, 0:p]
        self.new_coordinates = np.dot(self.new_bases.T, self.images)
        return self.new_coordinates.T

    def show_eigen_face(self, height, width, min_pix_int, max_pix_int, eig_no):
        ev = self.new_bases[:, eig_no:eig_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)
        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
        ev_re = np.reshape(ev, (height, width))
        cv2.imshow("Eigen Face " + str(eig_no),  cv2.resize(np.array(ev_re, dtype = np.uint8),(200, 200)))
        cv2.waitKey()
        
    def new_cord(self, name, img_height, img_width):
        img = cv2.imread(name)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        img_vec = np.asmatrix(gray).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.img_labels)) + img_vec)/(len(self.img_labels) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)
  
    def new_cord_for_image(self, image):
        img_vec = np.asmatrix(image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.img_labels)) + img_vec) / (len(self.img_labels) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def recognize_face(self, new_cord_pca, k=0):
        classes = len(self.no_of_elements)
        start = 0
        distances = []
        for i in range(classes):
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.no_of_elements[i])]
            mean_temp = np.mean(temp_imgs, 1)
            start = start + self.no_of_elements[i]
            dist = s_linalg.norm(new_cord_pca - mean_temp)
            distances += [dist]
           
           
        min = np.argmin(distances)
        threshold = 100000
        if distances[min] < threshold:
            print("Person", k, ":", min, self.img_targets[min])
            return self.img_targets[min]
        else:
            print("Person", k, ":", min, 'Unknown')
            return 'Unknown'




        

    def img_from_path(self, path , img_width,img_height):
        gray = cv2.imread(path, 0)
        return cv2.resize(gray , (img_width , img_height))


# optional Functions


    def show_image(self, label_to_show, img):

        cv2.imshow(label_to_show, img)
        cv2.waitKey()
    