def renameImgName(dirPath):
    files = os.listdir(dirPath)
    for file in files:
        oldPath = os.path.join(dirPath, file)
        newPath = os.path.join(dirPath, file.split('-')[0] + '.png')
        if not os.path.exists(newPath):
            os.rename(oldPath, newPath)


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




if __name__ == '__main__':
    renameImgName()
