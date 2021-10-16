#https://qiita.com/Hironsan/items/56f6c0b2f4cfd28dd906
#を参考に試作


#import list

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
        "R1", # 7 ロボット1
        "R2", # 8 ロボット2
        "R3", # 9 ロボット3
        "R3", # 10 ロボット4
        "B1", # 11 基地局１
        "B2", # 12 基地局2
        "B3", # 13 基地局3
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
        for i in range(4):
            name = "G" + str(i)
            tmp = self._find_pos(name)[0]
            self.goal_list.append(tmp)
        
        self.done = False
        self.damage = 0
        self.steps = 0
        return self._observe()
    


    # actionをmapに反映させて反映後のmapと報酬と終了したか否かを返す。
    def step(self,action):
        #移動action　0.右right 1.上up 2.左left 3.下down
        if action == 0:
            next_pos = self.pos + [0, 1]
        elif action == 1:
            next_pos = self.pos + [-1, 0]
        elif action == 2:
            next_pos = self.pos + [0, -1]
        elif action == 3:
            next_pos = self.pos + [1, 0]

        if self._is_movable(next_pos):
            self.pos = next_pos
            moved = True
        else:
            moved = False

        observation = self._observe()
        reward = self._get_reward(self.pos, moved)
        self.damage += self._get_damage()
        self.done = self._is_done()
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
        if moved and (self.goal == pos).all():
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
        )

    def _observe(self):
        # マップにロボットの位置を重ねて返す
        observation = self.MAP.copy()
        observation[tuple(self.pos)] = self.FIELD_TYPES.index('R1')
        return observation

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if (self.pos == self.goal).all():
            return True
        elif self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
        self.MAP == self.FIELD_TYPES.index(field_type)
    ))))
    
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
        self.selected_AP = None
        self.steps = 0
        
        #周囲の環境　4dは４方向で最も距離が近い障害までの距離　8mは周囲８マスに障害があるかどうか
        self.surround_4d = [100]*4
        self.surround_8m = [0]*8

    def _step(self,action,pos):
        self.steps+=1
        
        pass


    def _set_base(self,base):
        self.selected_base = base
        return
    
    def _is_done(self):
        pass

    def _is_loading(self):
        pass

    # observationに応じてactionを返す
    def get_action(self, observation):
        return 1


#アクセスポイントAP
class AP():
    #インスタンス変数 instance variable

    #初期化initialize
    def __init__(self,name,start_pos):
        self.name = name
        self.pos = start_pos
        self._reset()
    #メソッドmethods
    def _reset(self):
        pass
    
    def _step(self):
        pass


    def set_pos(self,pos):
        self.pos = pos
        return

    def set_selected_robot(self,robot):
        self.selected_robot.append(robot)
        return

#商品goods
class Goods():
    pass

#その他の関数

#動かすもの

def main():
    env=Environment()
    start_positions = env.get_start_positions()#スタートポジションはenvで定義しておく]　mainで作っても良い <= バラバラで想定
    names = env.get_names() # nameのリスト mainで定義しておいても良い
    state = env.get_state()# 初期のmap取得
    robots = [ Robot(names[i],start_positions[i]) for i in range(len(names))] #robotの生成

    while True:
        actions = [ r.get_action(state) for r in robots ] #actionlistを作る
        
        for i in range(env.get_robot_num()):
            env.set_action(i,actions) #robotのindex // ここではreward返したらだめ actionによってfieldを更新するだけ
        for i in range(env.get_robot_num()):
            robots.set_reward(env.get_reward(i)) #ロボットに報酬をセットする
        
        if env.is_end():# 終了したかどうか
            break

        state = env.get_state() #stateを更新する



# test用
def test():
    env=Environment()
    steplist=[0,0,0,0,0,1,1,1,1]

    count=1
    for i in range(len(steplist)):
        print("========================")
        print(count)
        count+=1
        print(env._step(steplist[i]))
        


    R1=Robot("Robot1",(1,1))
    R2=Robot("Robot2",(1,10))
    B1=AP("Base1",(0,3))
    B2=AP("Base2",(11,3))
    print(R1.name)
    print(env.goal)
    print(env.goal_list)



if __name__ == '__main__':
    main()



