import itchat
import time
import datetime
from itchat.content import TEXT

def get_friends():
    
    friends = itchat.get_friends(update=True)[1:]

    return friends


def get_number_of_friends(friends):

    num = 0

    for f in friends:
        num += 1

    return num

def get_gender_of_friends(friends):

    Sex = {'male':0, 'female':0, 'other':0}

    for f in friends:

        if f['Sex'] == 1:
            Sex['male'] += 1
        elif f['Sex'] == 2 :
            Sex['female'] += 1
        else:
            Sex['other'] += 1
    return Sex

def get_location_of_friends(friends):

    Location = []

    for f in friends:
        if f['City']:
             Location.append(f['City'])
        elif f['Province']:
            Location.append(f['Province'])
        else:
            Location.append(None)

    return Location


def get_chatroom_list():
    roomslist = itchat.get_chatrooms()
    return roomslist

def get_chatroom_information(roomName):
    itchat.dump_login_status()
    myroom = itchat.search_chatrooms(userName=roomName)
    return myroom

# 下面的源代码来自于：https://www.cnblogs.com/laoyueblogs/p/9975323.html

def getroom_message(n):
    #获取群的username，对群成员进行分析需要用到
    itchat.dump_login_status() # 显示所有的群聊信息，默认是返回保存到通讯录中的群聊
    RoomList =  itchat.search_chatrooms(name=n)
    if RoomList is None:
        pass
        #print("{0} group is not found!".format(name))
    else:
       # print('取得：',RoomList[0]['UserName'])
        return RoomList[0]['UserName']


def getchatrooms():
    #获取群聊列表
    roomslist = itchat.get_chatrooms()
    #print('列表',roomslist)
    return roomslist


if __name__ == '__main__':

    itchat.auto_login(hotReload=True)
    roomslist = []

    # friends = get_friends()
    # Sex = get_gender_of_friends(friends)
    # Location = get_location_of_friends(friends)
    # chatroom = get_chatroom_list()

    # print(type(friends))
    # num = get_number_of_friends(friends)
    # print('You\'ve got:%d'% num)
    # print('male:%d\nfemale:%d\nother:%d' % (Sex['male'],Sex['female'],Sex['other']))

    print("程序开始：",datetime.datetime.now())
    for i in getchatrooms():
        roomslist.append(i['NickName'])

    with open('群用户名.txt', 'w', encoding='utf-8')as f:
        f.write("你一共加入了{0}群".format(str(len(roomslist))))
        for n in roomslist:
            ChatRoom = itchat.update_chatroom(getroom_message(n), detailedMember=True)
            # print(ChatRoom['MemberList'])
            f.write('\n\n------------------------------群名称：'+ChatRoom['NickName']+"该微信群一共有{0}个成员".format(str(len(ChatRoom['MemberList'])))+'----------------------------------\n')
            #print("ChatRoom",ChatRoom)
            for i in ChatRoom['MemberList']:
                f.write("昵称："+i['NickName']+'：性别：'+str(i['Sex'])+'：省份：'+i['Province']+'\n')
        # f.close()
        print("程序结束：",datetime.datetime.now())
    


    

