import os
import shutil


def renameImgName(dirPath):
    files = os.listdir(dirPath)
    for file in files:
        oldPath = os.path.join(dirPath, file)
        if len(file.split('-'))>1:
            newPath = os.path.join(dirPath, file.split('-')[0] + '.png')
            if not os.path.exists(newPath):
                os.rename(oldPath, newPath)

def rename(dirPath):
    files=os.listdir(dirPath)
    for file in files:
        oldPath = os.path.join(dirPath, file)



def filterLabel(imgdir, labeldir, labeltargetdir):
    imgfiles = os.listdir(imgdir)
    labelfiles = os.listdir(labeldir)
    for img in imgfiles:
        for label in labelfiles:
            if img.split('.')[0] == label.split('.')[0] or label.startswith(img.split('.')[0]):
                shutil.copyfile(os.path.join(labeldir, label), os.path.join(labeltargetdir, label))
                break


def whoisnotin(imgdir, labeldir):
    imgfiles = os.listdir(imgdir)
    labelfiles = os.listdir(labeldir)
    for img in imgfiles:
        flag = 0
        for label in labelfiles:
            if img.split('.')[0] == label.split('.')[0] or label.startswith(img.split('.')[0]):
                flag = 1
                break
        if flag == 0:
            return img


def countNum(imgdir, labeldir):
    imgfiles = os.listdir(imgdir)
    labelfiles = os.listdir(labeldir)
    print(imgfiles)
    print(len(imgfiles))
    print(labelfiles)
    print(len(labelfiles))
    imgmap = {}
    labelmap = {}
    for img in imgfiles:
        floorName = img.split('-')[0]
        if floorName in imgmap.keys():
            imgmap[floorName] = imgmap[floorName] + 1
        else:
            imgmap.setdefault(floorName, 1)
    for label in labelfiles:
        labelName = label.split('@')[0]
        if labelName in labelmap.keys():
            labelmap[labelName] = labelmap[labelName] + 1
        else:
            labelmap.setdefault(labelName, 1)
    print("label有但是img没有\n")
    for (labelName, count) in labelmap.items():

        imgcount = imgmap.get(labelName)
        if not imgcount == count:
            print(labelName, ' ', imgcount, ' ', count, '\n')
    print("img有但是label没有\n")
    for (ImgName, imgCount) in imgmap.items():
        labelcount = labelmap.get(ImgName)
        if not labelcount == imgCount:
            print(ImgName, ' ', imgCount, ' ', labelcount, '\n')


def renameImgByJson(imgDir, labelDir):
    imgfiles = os.listdir(imgDir)
    labelfiles = os.listdir(labelDir)
    for i in range(len(imgfiles)):
        if imgfiles[i].split('-')[0] in labelfiles[i]:
            # os.rename(os.path.join(imgDir, imgfiles[i]),
            #           os.path.join(imgDir, labelfiles[i].split('.')[0]+'.png'))
            shutil.move(os.path.join(imgDir, imgfiles[i]),
                         os.path.join("C:\\Users\\Pang Hao\\Desktop\\dataset2\\filtered",labelfiles[i].split('.')[0]+'.png'))

def removeDouble(fileDir):
    fileNameList=os.listdir(fileDir)
    print(fileNameList)
    for fileName in fileNameList:
        if fileName.split('.')[0].endswith("(2)"):
            newName=fileName.split('.')[0][:-3]+'.'+fileName.split('.')[-1]
            oldPath=os.path.join(fileDir,fileName)
            newPath=os.path.join(fileDir,newName)
            os.rename(oldPath,newPath)

def filterValidJson(imgDir,labelDir):
    imgfiles = os.listdir(imgDir)
    labelfiles = os.listdir(labelDir)
    for label in labelfiles:
        flag=0
        for img in imgfiles:
            if img.split('.')[0] == label.split('.')[0] or label.startswith(img.split('.')[0]):
                flag=1
                break
        if flag==0:
            print(label)



if __name__ == '__main__':
    # countNum("C:/Users/Pang Hao/Desktop/dfataset2/img",
    #          "C:/Users/Pang Hao/Desktop/dfataset2/json")
    # renameImgByJson("C:\\Users\\Pang Hao\\Desktop\\dataset2\\img",
    #                 "C:\\Users\\Pang Hao\\Desktop\\dataset2\\json")
    # removeDouble("C:\\Users\\Pang Hao\\Desktop\\dataset2\\filtered")
    # filterValidJson("C:\\Users\\Pang Hao\\Desktop\\dataset2\\filtered","C:\\Users\\Pang Hao\\Desktop\\dataset2\\json")
    # renameImgName(r'C:\Users\Pang Hao\Desktop\sjjdataset3\img')
    # whoisnotin(r'C:\Users\Pang Hao\Desktop\sjjdataset3\img',r'C:\Users\Pang Hao\Desktop\sjjdataset3\json')
    renameImgName(r"C:\Users\Pang Hao\Desktop\sjjdataset2\croped")
