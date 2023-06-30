import json
import shutil
import sys
import time
import os
import urllib.request
import selenium
from requests import JSONDecodeError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
import requests
import asyncio


def FetchJsonFromSJJ():
    cityNumbers = {
        # "上海": "310100",       done
        # "北京": "110100",      done
        "天津": "120100",
        # "哈尔滨":"230100",
        # "南京":"320100",
        # "广州":"440100",
        # "深圳": "440300",
        # "成都": "510100",
        # "杭州": "330100"

    }
    idList = []
    cookie="gr_user_id=6fb471bc-a2f0-4afd-bbc2-54b8b5d71844; t=2fef9c1972b617d0935de19507e61860; cna=w2jWHBoZ1ikCAZJGkjL74pMw; user=%7B%22avatar%22%3A%22%22%2C%22nickName%22%3A%22%E8%AE%BE%E8%AE%A1%E5%B8%882344%22%2C%22memberType%22%3A%22designer%22%2C%22memberId%22%3A%223233517015626891264%22%2C%22env%22%3A%22prod%22%7D; _gid=GA1.2.847912475.1687943017; xlly_s=1; cookie2=16c5e61f41abf18fdc0a41af14c00c53; _tb_token_=e3e117166166b; a0b8f1838a1126e3_gr_session_id=6e2362b8-f666-4694-b695-fe211fe627b3; a0b8f1838a1126e3_gr_session_id_sent_vst=6e2362b8-f666-4694-b695-fe211fe627b3; _m_h5_tk=b0c5d2f4f89861312fd301653af22ce4_1688122504206; _m_h5_tk_enc=0450104e5fb7f4f138fb1f530b1ba4d8; _gat=1; _ga_34B604LFFQ=GS1.1.1688113143.13.1.1688113145.58.0.0; _ga=GA1.1.124063153.1687312404; isg=BLCw7dleoC3pM3wlAvYl4Gu4gX4C-ZRDyAHL7qoBr4v3ZVIPUwiJ0ooXvW0FdUwb"

    for cityName, cityNum in cityNumbers.items():
        # 20页
        print("下载json开始!")
        for page in range(0, 6):
            print(f"当前下载epoch为:{page}")
            postBody = {"attributes": [f"{cityNum}"],
                        "cityId": f"{cityNum}",
                        "searchText": "",
                        "offset": page + 1,
                        "limit": 100,
                        "sortField": "time_created",
                        "sortOrder": "desc"}
            headers = {
                'Content-Type': 'application/json;charset=UTF-8',
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Host": "api.shejijia.com",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Cookie": f"{cookie}",
                "Referer": "https://www.shejijia.com/",
                "Origin": "https://www.shejijia.com",
                "Platform": "admin",
                "Sec-Ch-Ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                "Pragma": "no-cache",
                "Cache-Control": "no-cache",
                "Authority": "api.shejijia.com",
                "Sec-Ch-Ua-Platform": "Windows",
                "Accept-Language":"zh-CN,zh;q=0.9,zh-TW;q=0.8",
                "Sec-Ch-Ua-Mobile":"?0",
                "Sec-Fetch-Mode":"cors"
            }
            json_data = json.dumps(postBody)
            time.sleep(10)
            response = requests.post("https://api.shejijia.com/roommgr/api/rest/v1.0/templateasset/search",
                                     data=json_data,
                                     headers=headers)
            # print(response.content, response.status_code)  # 这里可以拿到json和有水印的img
            try:
                for i in response.json().get("data").get("data"):
                    idList.append(i.get("id"))
                    filePath = f"./data/json/{cityName}"
                    fileName = f"{i.get('name') + '@' + i.get('id') + '.json'}"
                    fileUrl = changeExt(i.get('meta').get('image'))

                    if not os.path.exists(filePath):
                        os.makedirs(filePath)
                    if not os.path.exists('./history'):
                        os.makedirs('./history')

                    if not checkStrInFile(i.get("name"), f'./history/originid.txt'):
                        downloadJson(fileUrl, os.path.join(filePath, fileName))
                        with open(f'./history/originid.txt', 'a+') as f:
                            # origin文件里每行存两个内容,id和 name,如果name冲突,同样不保存id
                            f.write(i.get("id") + '\t' + i.get("name") + '\n')
                            f.flush()
                            f.close()
                            # id    name的格式
            except (JSONDecodeError,TypeError) as e:
                # print(i)
                print(page)
                print(e)
                sys.exit(1)

    return idList


def changeExt(urlString):
    fileName = os.path.basename(urlString)
    ext = os.path.splitext(fileName)[-1][1:]
    fileNameWithJson = urlString.replace(ext, 'json')
    return fileNameWithJson


def downloadJson(url, fileName):
    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))
    if not os.path.exists(fileName):

        try:
            urllib.request.urlretrieve(url, fileName)
            print("json下载完成！")
        except urllib.error.URLError as e:
            print('下载失败!', e)


# async def getOriginFloorImgFromSSJv1():
#     with open('./stealth.min.js', mode='r') as f:
#         js = f.read()
#     chromeDownloadPath = "D:\\Coding\\Web\\IKEA\\FetchData\\data\\img"
#     if not os.path.exists(chromeDownloadPath):
#         os.makedirs(chromeDownloadPath)
#     userData = f"C:\\Users\\Pang Hao\\AppData\\Local\\Google\\Chrome\\User Data"
#     chromeOptions = Options()
#     chromeOptions.headless = True
#     chromeOptions.add_experimental_option('excludeSwitches', ['enable-automation'])
#     chromeOptions.add_argument("--disable-blink-features")
#     chromeOptions.add_argument("--headless")
#     chromeOptions.add_argument(f'--user-data-dir={userData}')
#     chromeOptions.add_argument("--disable-blink-features=AutomationControlled")
#     user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
#     chromeOptions.add_argument(f'user-agent={user_agent}')
#     chromeOptions.add_experimental_option("prefs", {
#         "download.default_directory": chromeDownloadPath,
#         "download.prompt_for_download": False,
#         "download.directory_upgrade": True,
#         "safebrowsing.enabled": True
#     })
#
#     driver = webdriver.Chrome(options=chromeOptions)
#
#     historyStack = set([])
#
#     driver.execute_cdp_cmd(
#         cmd_args={'source': js},
#         cmd="Page.addScriptToEvaluateOnNewDocument",
#     )
#
#     wait = WebDriverWait(driver, 10)
#
#     try:
#         driver.get("https://www.shejijia.com/huxing/?cityId=310100&cityName=%E4%B8%8A%E6%B5%B7%E5%9F%8E%E5%8C%BA")
#     finally:
#         # 处理验证码
#         # while driver.title == '验证码拦截':
#         #     try:
#         #         ele1 = WebDriverWait(driver, timeout=2).until(lambda e: driver.find_element(By.ID, "nc_1_n1z"))
#         #         print(ele1)
#         #         ele1.click()
#         #     except (NoSuchElementException, TimeoutException):
#         #         pass
#         #     try:
#         #         ele2 = WebDriverWait(driver, timeout=2).until(
#         #             lambda ele: driver.find_element(By.CLASS_NAME, "errloading"))
#         #         print(ele2)
#         #         ele2.click()
#         #     except (NoSuchElementException, TimeoutException):
#         #         pass
#         #     try:
#         #         ele = WebDriverWait(driver, 10).until(
#         #             lambda e: driver.find_element(By.ID, "nc_1_n1z"))
#         #         driver.implicitly_wait(2)
#         #         actions = ActionChains(driver)
#         #         actions.click_and_hold(ele)
#         #
#         #         actions.move_by_offset(300, 0)
#         #         # actions.pause(duration/steps)
#         #         actions.release()
#         #         actions.perform()
#         #         driver.implicitly_wait(5)
#         #     except Exception:
#         #         pass
#         AuthCheck(driver)
#         print('当前页面标题:', driver.title)
#         print('当前页面URL:', driver.current_url)
#
#         # await login(driver)
#         historyStack.update(driver.window_handles)
#
#         lookMore = wait.until(
#             lambda e: driver.find_element(By.CSS_SELECTOR, ".floorplan .left a"))
#
#         lookMore.click()
#
#         # 这里打开了了一个新标签,切换标签页
#         AuthCheck(driver)
#
#         updateHistoryStack(driver, historyStack)
#         originalWindow = driver.current_window_handle
#         # print(driver.current_window_handle)
#         nums = len(
#             WebDriverWait(driver, timeout=10).until(
#                 lambda e: driver.find_elements(By.CSS_SELECTOR, ".floorplan .item>.card-container>.img-container>img")))
#
#         skip = 12
#         while True:
#             # 在这里加上跳页
#
#             if skip > 0:
#                 for i in range(skip):
#                     time.sleep(1)
#                     try:
#                         nextbutton = WebDriverWait(driver, timeout=15).until(
#                             EC.element_to_be_clickable((By.XPATH, "//button[@class='btn-next']")))
#                         AuthCheck(driver)
#                         nextbutton.click()
#                         AuthCheck(driver)
#                     except (TimeoutException, NoSuchElementException, ElementClickInterceptedException) as e:
#                         time.sleep(2)
#                         nextbutton = WebDriverWait(driver, timeout=10).until(
#                             EC.element_to_be_clickable((By.XPATH, "//button[@class='btn-next']")))
#                         AuthCheck(driver)
#                         nextbutton.click()
#                         AuthCheck(driver)
#                         pass
#                 skip = 0
#
#             for i in range(nums):
#                 time.sleep(2)
#                 # 这里直接点去设计
#                 designImgs = WebDriverWait(driver, timeout=10).until(
#                     lambda driver: driver.find_elements(By.CSS_SELECTOR,
#                                                         ".floorplan .item>.card-container>.img-container>img"))
#                 mouseMove = ActionChains(driver)
#
#                 mouseMove.move_to_element_with_offset(designImgs[i], xoffset=0, yoffset=115)
#                 time.sleep(2)
#                 mouseMove.perform()
#
#                 time.sleep(2)
#                 mouseMove.click().perform()
#                 time.sleep(2)
#
#                 # hrefs = WebDriverWait(driver, timeout=10).until(
#                 #     lambda e: driver.find_elements(By.CSS_SELECTOR, ".floorplan .info > a"))
#                 # driver.execute_script("arguments[0].click();",hrefs[i])
#                 updateHistoryStack(driver, historyStack)
#
#                 # 去设计按钮
#                 # goDetail = WebDriverWait(driver, timeout=10).until(
#                 #     lambda el: driver.find_element(By.CSS_SELECTOR, ".detail .info .btn"))
#                 # goDetail.click()
#                 AuthCheck(driver)
#                 print(driver.current_window_handle)
#                 updateHistoryStack(driver, historyStack)
#
#                 # login
#                 try:
#                     # loginButton = WebDriverWait(driver,10).until(lambda driver: driver.find_element(By.CSS_SELECTOR, "div.right:first-child"))
#                     # loginButton.click()
#
#                     # 这个元素是在iframe里面的,先进入iframe
#                     loginFrame = WebDriverWait(driver, 10).until(lambda d: d.find_element(By.ID, "alibaba-login-box"))
#                     currentPageId = driver.current_window_handle
#                     driver.switch_to.frame(loginFrame)
#                     # 进入密码登录
#                     passwordLogin = WebDriverWait(driver, 10).until(
#                         lambda driver: driver.find_element(By.CSS_SELECTOR, "i.iconfont"))
#                     passwordLogin.click()
#                     userName = WebDriverWait(driver, 10).until(lambda driver: driver.find_element(By.ID, "fm-login-id"))
#                     userName.send_keys("13343352074")
#                     password = WebDriverWait(driver, 10).until(
#                         lambda driver: driver.find_element(By.ID, "fm-login-password"))
#                     password.send_keys("ph177665ph")
#                     loginButton = WebDriverWait(driver, 10).until(
#                         lambda driver: driver.find_element(By.CSS_SELECTOR, ".fm-button.fm-submit.password-login"))
#                     loginButton.click()
#
#                     driver.switch_to.default_content()
#                 except (TimeoutException, NoSuchElementException) as e:
#                     pass
#
#                 time.sleep(2)
#                 # 3d设计
#                 try:
#                     toolBar = WebDriverWait(driver, timeout=30).until(
#                         lambda driver: driver.find_element(By.CSS_SELECTOR, ".toolbar ul.toollist"))
#                 except TimeoutException as e:
#                     time.sleep(20)
#
#                 # 提示在设计中,
#                 try:
#                     des = WebDriverWait(driver, timeout=4).until(lambda driver: driver.find_element(By.CSS_SELECTOR,
#                                                                                                     "div.auth-popup-container-content-action > div.auth-popup-container-content-action-btn.auth-popup-container-content-action-next > div.hs-iconfont-view > div > span"))
#                     des.click()
#                 except (TimeoutException, NoSuchElementException) as e:
#                     pass
#                 # 新手引导
#                 try:
#                     closeGuide = driver.find_element(By.CLASS_NAME, "guidebg-close-btn")
#                     closeGuide.click()
#                 except (NoSuchElementException, TimeoutException) as e:
#                     pass
#
#                 # 保存方案
#                 try:
#                     save = WebDriverWait(driver, timeout=30).until(
#                         EC.element_to_be_clickable((By.XPATH, f"//li[@data-toolbar-path=\"toolBar_save\"]")))
#                     save.click()
#                     time.sleep(3)
#                 except (TimeoutException, NoSuchElementException) as e:
#                     time.sleep(10)
#                     save = WebDriverWait(driver, timeout=30).until(
#                         EC.element_to_be_clickable((By.XPATH, f"//li[@data-toolbar-path=\"toolBar_save\"]")))
#                     save.click()
#                     time.sleep(3)
#
#                 # 点击下载,这里不能直接点击,必须鼠标一步一步hover后才能点
#
#                 try:
#                     WebDriverWait(driver, timeout=30).until(
#                         EC.element_to_be_clickable((By.CSS_SELECTOR, ".toolbar ul.toollist")))
#                     act = ActionChains(driver)
#                     time.sleep(2)
#                     act.move_to_element(driver.find_element(By.XPATH,
#                                                             "//span[contains(text(),'图纸&清单') and @class='topLevelLabel']"))
#                     act.click().perform()
#
#                     time.sleep(2)
#                     WebDriverWait(driver, timeout=30).until(
#                         EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'彩图导出')]")))
#                     act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'彩图导出')]")).perform()
#
#                     time.sleep(2)
#                     WebDriverWait(driver, timeout=30).until(
#                         EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'不带家具')]")))
#                     act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'不带家具')]")).perform()
#                     act.click().perform()
#                 except (TimeoutException, NoSuchElementException) as e:
#                     time.sleep(2)
#                     WebDriverWait(driver, timeout=30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".toollist")))
#                     act = ActionChains(driver)
#                     time.sleep(2)
#                     act.move_to_element(driver.find_element(By.XPATH,
#                                                             "//span[contains(text(),'图纸&清单') and @class='topLevelLabel']"))
#                     act.click().perform()
#
#                     time.sleep(2)
#                     WebDriverWait(driver, timeout=30).until(
#                         EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'彩图导出')]")))
#                     act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'彩图导出')]")).perform()
#
#                     time.sleep(2)
#                     WebDriverWait(driver, timeout=30).until(
#                         EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'不带家具')]")))
#                     act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'不带家具')]")).perform()
#                     act.click().perform()
#
#                     pass
#
#                 # exportKey=WebDriverWait(driver,10).until(lambda driver:driver.find_element(By.CSS_SELECTOR,"#floorplannerToolbar > div > ul > li.toolitem.toolBar_export_floorplan > div > div.dropmenus > ul > li:nth-child(2) > div > div.dropmenus.twolevelfolder > ul > li:nth-child(1)"))
#                 # exportKey.click()
#                 # print(href)
#
#                 # try:
#                 #     WebDriverWait(driver,timeout=10).until(EC.element_to_be_clickable((By.XPATH,'//span[contains(text(),"去保存")]')))
#                 #     goSave = WebDriverWait(driver, timeout=5).until(lambda driver: driver.find_element(By.XPATH,'//span[contains(text(),"去保存")]'))
#                 #     goSave.click()
#                 # except(TimeoutException) as e:
#                 #     pass
#                 # # WebDriverWait(driver, timeout=10).until(lambda driver: driver.find_element(By.CSS_SELECTOR,
#                 # #                                                                            "div.cad-export-dialog.homestyler-modal > div"))
#                 # try:
#                 #     WebDriverWait(driver,timeout=5).until(EC.visibility_of_element_located((By.CSS_SELECTOR,"div.cad-export-dialog.homestyler-modal > div > div")))
#                 #     time.sleep(2)
#                 #     WebDriverWait(driver,timeout=10).until(lambda driver:driver.find_element(By.XPATH,"//span[contains(text(),'jpg')]"))
#                 #     checkBox = WebDriverWait(driver, timeout=10).until(EC.element_to_be_clickable((By.XPATH,"//span[contains(text(),'jpg')]")))
#                 #     checkBox.click()
#                 #
#                 # except (ElementClickInterceptedException) as e:
#                 #     time.sleep(1)
#                 #
#                 #     checkBox = WebDriverWait(driver, timeout=10).until(EC.element_to_be_clickable((By.XPATH,"//span[contains(text(),'jpg')]")))
#                 #     checkBox.click()
#                 #     pass
#                 # expImg = WebDriverWait(driver, timeout=10).until(lambda driver: driver.find_element(By.CSS_SELECTOR,
#                 #                                                                                "div.cad-export-dialog__footer__right > button"))
#                 # # 点击下载
#                 # expImg.click()
#                 time.sleep(10)
#
#                 # driver.implicitly_wait(5)
#                 historyStack.remove(driver.current_window_handle)
#                 driver.close()
#                 driver.switch_to.window(originalWindow)
#                 # driver.back()
#                 AuthCheck(driver)
#
#             nextPage = WebDriverWait(driver, timeout=10).until(
#                 lambda driver: driver.find_element(By.CLASS_NAME, "btn-next"))
#             if nextPage.is_enabled():
#                 nextPage.click()
#                 AuthCheck(driver)
#             else:
#                 break
#
#         driver.close()
#
#         # 进入户型库


def updateHistoryStack(driver, historyStack):
    """
    a标签跳转后,打开的页面一定发生了变化,driver.window_handles也就变了
    :param driver:
    :param historyStack:
    :return:
    """
    # 打开的页面
    for window in driver.window_handles:
        if window not in historyStack:
            driver.switch_to.window(window)
            break
    historyStack.update(driver.window_handles)


def AuthCheck(driver):
    try:

        while driver.title == '验证码拦截' or bool(WebDriverWait(driver, timeout=2).until(
                EC.visibility_of_element_located((By.XPATH, "//iframe[@id=\"baxia-dialog-content\"]")))):
            if bool(WebDriverWait(driver, timeout=5).until(
                    EC.visibility_of_element_located((By.XPATH, "//iframe[@id=\"baxia-dialog-content\"]")))):
                checkframe = WebDriverWait(driver, timeout=5).until(
                    EC.visibility_of_element_located((By.XPATH, "//iframe[@id=\"baxia-dialog-content\"]")))
                driver.switch_to.frame(checkframe)
                flag = 1
                pass
            try:
                ele1 = WebDriverWait(driver, timeout=5).until(lambda e: driver.find_element(By.ID, "nc_1_n1z"))
                print(ele1)
                ele1.click()
            except (NoSuchElementException, TimeoutException):
                pass
            try:
                ele2 = WebDriverWait(driver, timeout=2).until(
                    lambda ele: driver.find_element(By.CLASS_NAME, "errloading"))
                print(ele2)
                ele2.click()
            except (NoSuchElementException, TimeoutException):
                pass
            try:
                ele = WebDriverWait(driver, 10).until(
                    lambda e: driver.find_element(By.ID, "nc_1_n1z"))
                driver.implicitly_wait(2)
                actions = ActionChains(driver)
                actions.click_and_hold(ele)

                actions.move_by_offset(300, 0)
                # actions.pause(duration/steps)
                actions.release()
                actions.perform()
                driver.implicitly_wait(5)

                driver.switch_to.default_content()

            except Exception:
                pass
    except (NoSuchElementException, TimeoutException) as e:
        pass


def checkStrInFile(checkstr, filePath):
    if not os.path.exists(filePath):
        file=open(filePath,"w")
        file.close()
    with open(filePath, "r") as f:
        for line in f.readlines():
            if checkstr.strip() == line.strip().split('\t')[-1]:
                return True
    return False


# 直接拿接口
def getOriginFloorImgFromSSJv2(init=True):
    if init:
        FetchJsonFromSJJ()

    origin = open('./history/originid.txt', 'r')
    originInfoList = origin.readlines()
    finished = open('./history/finish.txt', 'r')
    finishIdList = finished.readlines()
    jsHankFile = open('./stealth.min.js', mode='r')
    js = jsHankFile.read()
    #   图片保存路径(绝对路径)
    chromeDownloadPath = "C:\\Users\\Pang Hao\\Downloads"
    if not os.path.exists(chromeDownloadPath):
        os.makedirs(chromeDownloadPath)
    userData = f"C:\\Users\\Pang Hao\\AppData\\Local\\Google\\Chrome\\User Data"
    chromeOptions = Options()

    chromeOptions.add_experimental_option('excludeSwitches', ['enable-automation'])
    chromeOptions.add_argument("--disable-blink-features")
    # chromeOptions.add_argument(f'--user-data-dir={userData}')
    chromeOptions.add_argument("--disable-blink-features=AutomationControlled")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
    chromeOptions.add_argument(f'user-agent={user_agent}')
    chromeOptions.add_experimental_option("prefs", {
        "download.default_directory": chromeDownloadPath,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    chromeOptions.add_argument("--headless")
    driver = webdriver.Chrome(options=chromeOptions)
    driver.execute_cdp_cmd(
        cmd_args={'source': js},
        cmd="Page.addScriptToEvaluateOnNewDocument",
    )
    for info in originInfoList:  # 待下载户型id
        if checkStrInFile(info.strip().split('\t')[0], './history/finish.txt'):  # 已完成的户型id
            continue
        else:
            print("当前任务为: ",info.strip())
            try:
                assetId=info.strip().split('\t')[0]
                driver.get(f"https://3d.shejijia.com/?designType=publicdesign&assetId={assetId}")
            finally:
                AuthCheck(driver)

                print('当前页面标题:', driver.title)
                print('当前页面URL:', driver.current_url)

                # login
                try:
                    # loginButton = WebDriverWait(driver,10).until(lambda driver: driver.find_element(By.CSS_SELECTOR, "div.right:first-child"))
                    # loginButton.click()

                    # 这个元素是在iframe里面的,先进入iframe
                    loginFrame = WebDriverWait(driver, 10).until(lambda d: d.find_element(By.ID, "alibaba-login-box"))
                    currentPageId = driver.current_window_handle
                    driver.switch_to.frame(loginFrame)
                    # 进入密码登录
                    passwordLogin = WebDriverWait(driver, 10).until(
                        lambda driver: driver.find_element(By.CSS_SELECTOR, "i.iconfont"))
                    passwordLogin.click()
                    userName = WebDriverWait(driver, 10).until(lambda driver: driver.find_element(By.ID, "fm-login-id"))
                    userName.send_keys("13343352074")
                    password = WebDriverWait(driver, 10).until(
                        lambda driver: driver.find_element(By.ID, "fm-login-password"))
                    password.send_keys("ph177665ph")
                    loginButton = WebDriverWait(driver, 10).until(
                        lambda driver: driver.find_element(By.CSS_SELECTOR, ".fm-button.fm-submit.password-login"))
                    loginButton.click()

                    driver.switch_to.default_content()
                except (TimeoutException, NoSuchElementException) as e:
                    pass

                time.sleep(2)
                # 3d设计
                try:
                    toolBar = WebDriverWait(driver, timeout=30).until(
                        lambda driver: driver.find_element(By.CSS_SELECTOR, ".toolbar ul.toollist"))
                except TimeoutException as e:
                    pass

                # 提示在设计中,
                try:
                    des = WebDriverWait(driver, timeout=10).until(lambda driver: driver.find_element(By.CSS_SELECTOR,
                                                                                                     "div.auth-popup-container-content-action > div.auth-popup-container-content-action-btn.auth-popup-container-content-action-next > div.hs-iconfont-view > div > span"))
                    des.click()
                except (TimeoutException, NoSuchElementException) as e:
                    pass
                # 新手引导
                try:
                    closeGuide = WebDriverWait(driver, timeout=60).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "guidebg-close-btn")))

                    closeGuide.click()
                except (NoSuchElementException, TimeoutException) as e:
                    pass

                # 保存方案,这一步等久一点
                try:
                    save = WebDriverWait(driver, timeout=60).until(
                        EC.element_to_be_clickable((By.XPATH, f"//li[@data-toolbar-path=\"toolBar_save\"]")))
                    save.click()
                    time.sleep(5)
                except (TimeoutException, NoSuchElementException) as e:
                    pass

                # 点击下载,这里不能直接点击,必须鼠标一步一步hover后才能点

                try:
                    WebDriverWait(driver, timeout=120).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".toolbar ul.toollist")))
                    act = ActionChains(driver)
                    time.sleep(2)
                    act.move_to_element(driver.find_element(By.XPATH,
                                                            "//span[contains(text(),'图纸&清单') and @class='topLevelLabel']"))
                    act.click().perform()

                    # time.sleep(2)
                    WebDriverWait(driver, timeout=60).until(
                        EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'彩图导出')]")))
                    act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'彩图导出')]")).perform()

                    time.sleep(2)
                    WebDriverWait(driver, timeout=60).until(
                        EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'不带家具')]")))
                    act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'不带家具')]")).perform()
                    act.click().perform()
                except (TimeoutException, NoSuchElementException) as e:
                    WebDriverWait(driver, timeout=120).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, ".toolbar ul.toollist")))
                    act = ActionChains(driver)
                    time.sleep(2)
                    act.move_to_element(driver.find_element(By.XPATH,
                                                            "//span[contains(text(),'图纸&清单') and @class='topLevelLabel']"))
                    act.click().perform()

                    time.sleep(2)
                    WebDriverWait(driver, timeout=60).until(
                        EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'彩图导出')]")))
                    act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'彩图导出')]")).perform()

                    time.sleep(2)
                    WebDriverWait(driver, timeout=60).until(
                        EC.visibility_of_element_located((By.XPATH, "//span[contains(text(),'不带家具')]")))
                    act.move_to_element(driver.find_element(By.XPATH, "//span[contains(text(),'不带家具')]")).perform()
                    act.click().perform()

                time.sleep(20)
                with open("./history/finish.txt", "a+") as finish:
                    finish.write(f"{info}")
                    finish.flush()
                    finish.close()
                # driver.close()
                print("图片下载完成",info.strip(),'\n')
    finished.close()
    origin.close()
    jsHankFile.close()


if __name__ == "__main__":
    # getOriginFloorImgFromSSJv2(True)
    FetchJsonFromSJJ()
