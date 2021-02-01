import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The Implementation of Graph Pooling and Unpooling for NTU-RGB+D.
'''

class Down_Joint2Part(nn.Module):
    def __init__(self):
        super().__init__()
        self.torso = [0,1,20]
        self.left_leg_up = [16,17]
        self.left_leg_down = [18,19]
        self.right_leg_up = [12,13]
        self.right_leg_down = [14,15]
        self.head = [2,3]
        self.left_arm_up = [8,9]
        self.left_arm_down = [10,11,23,24]
        self.right_arm_up = [4,5]
        self.right_arm_down = [6,7,21,22]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 3))                                             
        x_leftlegup = F.avg_pool2d(x[:, :, :, self.left_leg_up], kernel_size=(1, 2))                               
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                    
        x_rightlegup = F.avg_pool2d(x[:, :, :, self.right_leg_up], kernel_size=(1, 2))                       
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                  
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 2))                                    
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                    
        x_leftarmdown = F.avg_pool2d(x[:, :, :, self.left_arm_down], kernel_size=(1, 4))              
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                
        x_rightarmdown = F.avg_pool2d(x[:, :, :, self.right_arm_down], kernel_size=(1, 4))            
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)            
        return x_part


class Down_Part2Body(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))                                           
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 2))                          
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 2))                       
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 2))                           
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 2))                       
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)              
        return x_body

class Down_Body2Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.node = [0,1,2,3,4]
    def forward(self, x):
        x_node = F.avg_pool2d(x[:,:,:,self.node], kernel_size=(1, 5))
        return x_node


class Up_Part2Joint(nn.Module):

    def __init__(self):
        super().__init__()
        # for all: index - 1
        self.torso = [0,1,20]
        self.left_leg_up = [16,17]
        self.left_leg_down = [18,19]
        self.right_leg_up = [12,13]
        self.right_leg_down = [14,15]
        self.head = [2,3]
        self.left_arm_up = [8,9]
        self.left_arm_down = [10,11,23,24]
        self.right_arm_up = [4,5]
        self.right_arm_down = [6,7,21,22]

    def forward(self, part):
        N, d, T, w = part.size()  
        x = part.new_zeros((N, d, T, 25))

        x[:,:,:,self.left_leg_up] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = torch.cat((part[:,:,:,2].unsqueeze(-1), part[:,:,:,2].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,6].unsqueeze(-1),part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = torch.cat((part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,8].unsqueeze(-1),part[:,:,:,8].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = torch.cat((part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1)),-1)

        return x


class Up_Body2Part(nn.Module):

    def __init__(self):
        super().__init__()

        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]

    def forward(self, body):
        N, d, T, w = body.size()
        x = body.new_zeros((N, d, T, 10))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x


class up_Node2Body(nn.Module):
    def __init__(self):
        super().__init__()

        self.node = [0,1,2,3,4]

    def forward(self, node):
        N, d, T, w = node.size()
        x = node.new_zeros((N, d, T, 5))

        x[:,:,:,self.node] = torch.cat((node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1]),-1)

        return x