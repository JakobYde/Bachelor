from numpy import mean
from ImageInformation import ImageInformation
import csv

class Session(object):  
    jointNames = ['MCP1_PD', 'MCP2_PD', 'MCP3_PD', 'MCP4_PD', 'MCP5_PD', 'PIP1_PD', 'PIP2_PD', 'PIP3_PD', 'PIP4_PD', 'PIP5_PD']


    def __init__(self):
        self.files = []
        self.joints = []
        self.files
        for joint in Session.jointNames:
            img = ImageInformation()
            img.joint = joint
            self.files.append(img)
        self.avg = 0
        self.CRP = 0
        self.DAS28 = 0

    def __str__(self):
        return [str(file.score) for file in self.files]

    def calc_avg(self):
        self.avg = mean([f.score for f in self.files])

    def add_file(self, filename, joint, score):
        img = ImageInformation()
        img.fileName = filename
        img.joint = joint
        img.score = score
        self.joints.append(joint)
        self.files[Session.jointNames.index(joint)] = img

    def contains_duplicates(self):
        for i, joint in enumerate(self.joints):
            if joint in self.joints[i + 1:]:
                return True
        return False
    
    def missing_data(self):
        for joint in Session.jointNames:
            if joint not in self.joints:
                return True
        return False

    def no_problems(self):
        return not (self.contains_duplicates() or self.missing_data())

    def print_to_file(self, filename):
        status = [self.contains_duplicates(), self.missing_data(), self.no_problems()]
        with open(filename, mode='a') as file:
            writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(status + self.joints)
        return (status, len(self.joints))
