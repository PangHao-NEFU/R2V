import os
import ezdxf


class PreprocessCAD(object):

    def __init__(self, cadFolderPath: str):
        self.cadFolderPath = cadFolderPath
        self.currentCadFile = ''

    def analysisCadFile(self):
        cadFiles = os.listdir(self.cadFolderPath)
        for cadFile in cadFiles:
            if os.path.isfile(os.path.join(self.cadFolderPath,cadFile)):
                self.currentCadFile = os.path.join(self.cadFolderPath, cadFile)
                self.calcPoints(self.currentCadFile)


    def calcPoints(self, cadfilePath: str):
        try:
            currentDwg=ezdxf.readfile(self.currentCadFile)
            modelspace=currentDwg.modelspace()
            for entity in modelspace:
                print(entity)

        except OSError as e:
            print(f"read file error:{e}")
            pass

if __name__ == "__main__":
    cadFileDirPath=r"C:\Users\Pang Hao\Desktop\cad\cad"
    cadObject = PreprocessCAD(cadFileDirPath)
    cadObject.analysisCadFile()
