#https://qiita.com/Hironsan/items/56f6c0b2f4cfd28dd906
#を参考に試作


#import list

from math import sqrt
import numpy as np
import gym
import gym.spaces
import io
import sys
import copy
#環境Environment

class Environment():
    #インスタンス変数 instance variable
    metadata = {'render.modes': ['human', 'ansi']}
    FIELD_TYPES = [
        "S", # 0 スタート地点
        "P", # 1 plane 平面
        "O", # 2 obstacle 障害物
        "G0", # 3 ゴール地点
        "G1", # 4 ゴール地点
        "G2", # 5 ゴール地点
        "G3", # 6 ゴール地点
        "R0", # 7 ロボット0
        "R1", # 8 ロボット1
        "R2", # 9 ロボット2
        "R3", # 10 ロボット3
        "AP0", # 11 基地局0
        "AP1", # 12 基地局1
        "AP2", # 13 基地局2
    ]
    MAP = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  
        [2, 3, 1, 1, 4, 1, 1, 5, 1, 1, 6, 2],  
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  
        [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12],  
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 
        [2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2],  
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
    ])

    MAX_STEPS = 100

    #初期化initialize
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD_TYPES),
            shape=self.MAP.shape
        )
        self.reward_range = [-1., 100.]
        self._reset()
        pass
    
    #メソッドmethods
    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = self._find_pos('S')[0]
        self.goal = self._find_pos('G0')[0]
        
        #ゴール複数
        self.goal_list = list()
        for i in range(4):#４つで固定している
            name = "G" + str(i)
            tmp = self._find_pos(name)[0]
            self.goal_list.append(tmp)
        
        self.done = False

        self.damage = 0
        self.steps = 0
        return self._observe([self.pos])
    
    def _step(self,Robots,items):
        next_pos_list=[]
        for i in range(len(Robots)):
           tmp_pos = Robots[i]._step(items)
           next_pos_list.append(tmp_pos)

        #print(next_pos_list)

        observation = self._observe(next_pos_list)
        reward=0
        #reward = self._get_reward(self.pos, moved)
        #self.damage += self._get_damage()
        #self.done = self._is_done()
        return observation, reward, self.done, {}
        
    def _render(self,mode='human',close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n'.join(' '.join(
                self.FIELD_TYPES[elem] for elem in row
                ) for row in self._observe()
            ) + '\n'
        )
        return outfile
        
    def _close(self):
        pass
    def _seed(self,seed=None):
        pass

    #その他関数Others
    def _get_reward(self, pos, moved):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # - ゴールにたどり着くと 100 ポイント
        # - ダメージはゴール時にまとめて計算 ##特定地点ペナルティ
        # - 1ステップごとに-1ポイント(できるだけ短いステップでゴールにたどり着きたい)
        # とした
        if moved and (self.goal == pos):
            return max(100 - self.damage, 0)
        else:
            return -1
    
    #特定地点でペナルティ入れるなら使う
    def _get_damage(self):
        return 0
    
    def _is_movable(self, pos):
        # マップの中にいるか、歩けない場所にいないか
        return (
            0 <= pos[0] < self.MAP.shape[0]
            and 0 <= pos[1] < self.MAP.shape[1]
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'O'
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'AP0'
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'AP1'
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'AP2'
        )

    def _observe(self,poslist):
        # マップにロボットの位置を重ねて返す
        observation = self.MAP.copy()
        for i in range(len(poslist)):
            name = "R" + str(i)
            observation[tuple(poslist[i])] = self.FIELD_TYPES.index(name) 
        return observation

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if (self.pos == self.goal):
            return True
        elif self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
        self.MAP == self.FIELD_TYPES.index(field_type)
    ))))

    def cal_distance(self,pos1,pos2):
        distance = sqrt( (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 )
        return distance
    
    def cal_Receive_Power(self,d):
        '''
        入力
        Pt  送信電力(W)
        d 伝搬距離(m)
        Lambda 波長(m) = c/f = 299792458/f
        Gt　送信利得
        Gr　受信利得

        出力
        Pr　受信電力(W)
        '''
        Pt = 1
        Lambda = 299792458/(2.4*(10**9))
        Gt = 1
        Gr = 1

        Pr = Pt*Gt*Gr*((Lambda/(4*np.pi*d))**2)
        return Pr

#ロボットRobot
class Robot():
    #インスタンス変数 instance variable

    #初期化initialize
    def __init__(self,name,start_pos):
        self.name = name
        self.start_pos = start_pos
        self._reset()
        
    #メソッドmethods
    def _reset(self):
        self.pos = copy.copy(self.start_pos)
        self.goal = self.pos 
        self.selected_AP = None
        self.steps = 0
        self.action = 0
        self.flag_load = False #荷物をもっている状態True
        
        #周囲の環境　4dは４方向で最も距離が近い障害までの距離　8mは周囲８マスに障害があるかどうか
        self.surround_4d = [100]*4
        self.surround_8m = [0]*8

    def _step(self,items):
        #荷物の積み下ろしがあれば行う
        if (self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1]):
            #荷物を下ろす
            if self.flag_load:
                self.unload()
                
            #荷物を積む
            else:
                self.load(items)
        
        #移動方向の決定       
        self.action = self.cal_action(self.pos,self.goal)
        self.update_pos(self.action)
        return self.pos

    #行動の決め方（仮）4方向でゴールにより近づく方向に80%で進む
    def cal_action(self,pos,goal):
        env=Environment()
        d=env.cal_distance(pos,goal)
        distances=[]
        tmp=pos
        
        right=list(tmp)
        right[1]+=1
        
        d=env.cal_distance(right,goal)
        distances.append(d)
        
        up=list(tmp)
        up[0]-=1
        d=env.cal_distance(up,goal)
        distances.append(d)
        
        left=list(tmp)
        left[1]-=1
        d=env.cal_distance(left,goal)
        distances.append(d)
        
        down=list(tmp)
        down[0]+=1
        d=env.cal_distance(down,goal)
        distances.append(d)
        d_s = sorted(distances)
        pr = 0
        r=np.random.rand()
        if r<0.8:
            pr = 0
        else:
            pr = 1
            r=np.random.rand()
            if r>=0.8:
                pr = 2
        index=distances.index( (d_s[pr]) )
        action=index
        #荷物なくなって動きたくないとき
        if (pos[0]==goal[0] and pos[1]==goal[1]):
            action=-1
        return action

    #位置座標の更新
    def update_pos(self,action):
        env=Environment()
        if action != -1:
            next_pos = list(self.pos)
            if action == 0:
                next_pos[1] += 1
            elif action == 1:
                next_pos[0] -= 1
            elif action == 2:
                next_pos[1] -= 1
            elif action == 3:
                next_pos[0] += 1
            next_pos = tuple(next_pos)
            
            if env._is_movable(next_pos):
                self.pos = next_pos
                self.steps+=1
                moved = True
            else:
                moved = False
        
        else:
            moved = False
        

    def _set_base(self,base):
        self.selected_base = base
        return
    
    def _is_done(self):
        pass
    
    # 荷物を積む
    def load(self,items):
        env=Environment()
        if items.index<len(items.items_list):#まだ運ぶ荷物がある
            self.goal = env._find_pos("G" + str(items.items_list[items.index]))[0]
            self.flag_load = True
            items.index+=1
        else:#その場で待機させておく？
            pass
        if items.index==len(items.items_list):
            print("all items were carried")

    #荷物を下ろす     
    def unload(self):
        self.goal = self.start_pos #今はロボットの初期位置がスタートと同じ
        self.flag_load = False



#アクセスポイントAP
class AP():
    #インスタンス変数 instance variable

    #初期化initialize
    def __init__(self,name,start_pos):
        self.name = name
        self.start_pos = start_pos
        self._reset()
    #メソッドmethods
    def _reset(self):
        self.pos = self.start_pos
        
    
    def _step(self):
        pass


    def set_pos(self,pos):
        self.pos = pos
        return

    def set_selected_robot(self,robot):
        self.selected_robot.append(robot)
        return

#商品Items
class Items():
    def __init__(self,items):
        self.items_list = items
        self._reset()
    def _reset(self):
        self.index=0
    pass

#その他の関数

#動かすもの
env=Environment()

robot_names = ["R0","R1","R2","R3"]
robot_num=len(robot_names)
robot_start_positions = [(5,5)]*4
Robots = [Robot(robot_names[i],robot_start_positions[i]) for i in range(robot_num)]

ap_names = ["AP0","AP1"]
ap_num = len(ap_names)
ap_start_positions = [(3,0),(3,11)]
APs = [AP(ap_names[i],ap_start_positions[i]) for i in range(ap_num)]

itemlist = [0,1,2,3,0,2,3,0,0]

items=Items(itemlist)

for i in range(50):
    print(env._step(Robots,items))
    print("==============================")




