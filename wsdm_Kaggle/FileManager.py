import cPickle
import os
import os.path
import copy_reg
import types
import multiprocessing
import errno

#handling the pickling object
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


dir_path = os.path.dirname(os.path.realpath(__file__))  + "/"



class FileManager:
    """
        This allows for saving a ML model to be loaded at a later time.
    """

    def __init__( self, file_name ):
        self.file_name = file_name

        modelDir = dir_path + "models/"

        #check if directory exist, otherwise create the 
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)

        self.fileNameWithPath = modelDir + self.file_name



    def save (self, classifier):
        '''
            write to file
        '''
        with open(self.fileNameWithPath, 'wb') as fout:
            cPickle.dump(classifier, fout)
        print 'model written to: ' + self.file_name



    def load (self):
        """
            load an existing file
        """
        #ispresent =  os.path.exists(self.fileNameWithPath)
        ispresent =  os.path.isfile(self.fileNameWithPath)

        if ispresent:
            module = cPickle.load( open( self.fileNameWithPath ) ) #get the model
            return module

        print "path: " + self.fileNameWithPath +" flag: "+ str(ispresent)

        raise FileNotFoundError ("Attempting to load a module that was not saved!!!!")


    def isExist(self):
        #ispresent =  os.path.exists(self.fileNameWithPath)
        ispresent =  os.path.isfile(self.fileNameWithPath)
        return ispresent


if __name__ == "__main__":
    filename = "linear.pkl"
    fObject = FileManager(filename)

    fObject.save (mObject) #save a model

    model = fObject.load( ) #load a model









