class ImageInformation(object):
    def __init__(self):
        self.fileName = ""
        self.joint = ""
        self.score = -1
        
    def __repr__(self):
        return '(' + str(self.fileName) + ", " + str(self.joint) + ', ' + str(self.score) + ')'
    """description of class"""

    def __getfilename__(self):
        return self.fileName
