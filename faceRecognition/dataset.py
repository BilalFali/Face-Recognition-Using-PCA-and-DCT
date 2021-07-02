import os


class datasetClass:
    def __init__(self, required_no):

        self.dir = ("media/dataset")
        self.images_name_for_train = []
        self.target_name_as_array =[]
        self.target_name_as_set ={}
        self.labels_for_train = []
        self.no_of_elements_for_train = []

        self.images_name_for_test = []
        self.labels_for_test = []
        self.no_of_elements_for_test = []

        #self.images_targets = []

        person_no = 0

        for name in os.listdir(self.dir):
            dir_path = os.path.join(self.dir, name)
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no:
                    i = 0
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)

                        if i < required_no:
                            self.images_name_for_train += [img_path]
                            self.labels_for_train += [person_no]

                            if len(self.no_of_elements_for_train) > person_no:
                                self.no_of_elements_for_train[person_no] += 1
                            else:
                                self.no_of_elements_for_train += [1]
                            if i == 0:
                                self.target_name_as_array += [name]
                                self.target_name_as_set[person_no] = name


                        else:
                            self.images_name_for_test += [img_path]
                            self.labels_for_test += [person_no]

                            if len(self.no_of_elements_for_test) > person_no:
                                self.no_of_elements_for_test[person_no] += 1
                            else:
                                self.no_of_elements_for_test += [1]

                        i += 1
                    person_no += 1
