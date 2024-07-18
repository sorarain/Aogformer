import math
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers import RobertaForMaskedLM
from torch.optim import SGD


class AOG_Building_Block(nn.Module):
    # AOG_Building_Block
    # params： in_channels是输入特征图的通道数，out_channels输出特征图的通道数
    # sub_nums就是上述所说的word个数
    # sub_in_channels和sub_out_channels分别表示与一个单词相关的输入通道数和输出通道数
    # Ttype指定了T采用何种类型构造方式

    def __init__(self, num_blocks, config):
        super(AOG_Building_Block, self).__init__()
        self.attn = RobertaSelfAttention(config)
        self.linear = nn.Linear(config.hidden_size,config.hidden_size)
        # self.q = nn.Linear(config.hidden_size, 1)
        self.sub_nums = num_blocks
        self.config = config
        
        # # 调用construct函数（BFS算法）将构造图graph
        # self.structure = self.construct()
        # # 调用reconstruct函数（DFS算法）重构图graph
        # self.structure = self.reconstruct()
        # # 根据重构的图graph构造网络结构，用ModuleList存储网络结构，依次执行
    
    # BFS算法，用[]记录整张图中各个节点，
    # 每个节点用一个{}记录内容，其中各种节点包含的key如下：
    # or_node: type, start（文章中i）, end（文章中j）, children(多个index number的list)
    # and_node: type, left_start, left_end, 
    #           right_start, right_end, children（只有连个index number的list）
    # t_node: type, start, end
    # 每种节点还包括:in_channels(节点的输入特征通道数)
    #               out_channels（节点的输出特征通道数）
    #               history_key(节点的唯一标识，用于防止节点被重复构建)
    def construct(self):
        history_key = 'or_0_{}'.format(self.sub_nums-1)
        max_depth = math.ceil(math.log(self.sub_nums) / 2)
        max_num_nodes = self.sub_nums * 2
        structure = [{'type':'or', 'start':0, 'end':self.sub_nums-1,
                         'history_key':history_key,
                         'depth':0,
                         }]
        history = {history_key:0}
        
        sum_nodes = 1
        idx = 0
        # 利用循环进行构建，当图中所有的节点都遍历过时，说明graph已经构建完成。
        # or、and、t节点分别进行处理
        while(idx < sum_nodes):
            now_node = structure[idx]
            if now_node['type'] == 'or':
                nnum = now_node['end'] - now_node['start'] + 1
                history_key = 't_{}_{}'.format(now_node['start'], now_node['end'])
                if history_key in history:
                    now_node['children'] = [history[history_key]]
                else:
                    structure.append({'type':'t', 
                                     'start':now_node['start'], 
                                     'end':now_node['end'],
                                     'history_key':history_key,
                                     'depth':now_node['depth'],
                                     })
                    history[history_key] = sum_nodes
                    now_node['children'] = [sum_nodes]
                    sum_nodes += 1
                _d = list(range(math.floor(math.log2(now_node['end'] - now_node['start']))))
                _d = [2**x for x in _d]
                _d = [0] + _d
                d = [now_node['start'] + x for x in _d] #+ [now_node['end'] - x for x in _d]
                d = list(set(d))
                # for m in range(now_node['start'], now_node['end']):
                    # mid = m 
                for m in d:
                    mid = m
                    if m < now_node['start'] or m >= now_node['end'] or sum_nodes > max_num_nodes:
                        continue
                    history_key = 'and_{}_{}_{}'.format(now_node['start'], mid, 
                                                        now_node['end'])
                    if history_key in history:
                        now_node['children'].append(history[history_key])
                    else:
                        structure.append({'type':'and', 
                                          'left_start':now_node['start'],
                                          'left_end':mid,
                                          'right_start':mid+1, 
                                          'right_end':now_node['end'],
                                          'history_key':history_key,
                                          'depth':now_node['depth']+1,
                                          })
                        history[history_key] = sum_nodes            
                        now_node['children'].append(sum_nodes)
                        sum_nodes += 1
            elif now_node['type'] == 'and':
                left_history_key, left_child = \
                    self.get_andnode_child(now_node['left_start'],
                                           now_node['left_end'],
                                           now_node['depth'],
                                           force_t=now_node['depth'] == max_depth and max_num_nodes+2 < sum_nodes)
                right_history_key, right_child = \
                    self.get_andnode_child(now_node['right_start'], 
                                           now_node['right_end'],
                                           now_node['depth'],
                                           force_t=now_node['depth'] == max_depth and max_num_nodes+2 < sum_nodes)
                if left_history_key in history:
                    now_node['children'] = [history[left_history_key]]
                else:
                    structure.append(left_child)
                    now_node['children'] = [sum_nodes]
                    history[left_history_key] = sum_nodes
                    sum_nodes += 1
                
                if right_history_key in history:
                    now_node['children'].append(history[right_history_key])
                else:
                    structure.append(right_child)
                    now_node['children'].append(sum_nodes)
                    history[right_history_key] = sum_nodes
                    sum_nodes += 1 
            elif now_node['type'] == 't':
                pass
            idx += 1
        return structure
    
    # 根据start和end得到and节点不同的子节点
    # 如果start == end 将不需要Or节点（sub-sentence），否则需要Or节点并需要继续搜索
    def get_andnode_child(self, start, end, depth, force_t=False):
        nnum = end-start+1
        if start == end or force_t:
            key = 't_{}'.format(start) if start == end else 't_{}_{}'.format(start, end)
            temp = {'type':'t',
                    'start':start,'end':end,
                    'history_key':key,
                    'depth':depth,
                    }
            # if force_t:
            #     print(end - start,start, end)
        else:
            key = 'or_{}_{}'.format(start, end)
            temp = {'type':'or',
                    'start':start,'end':end,
                    'history_key':key,
                    'depth':depth+1}
        return key, temp    

    # DFS
    # 利用深度优先搜索对graph进行重构，节点的key没有发生变化
    # 由于一个Block的网络深度一般不会太深，且只在训练或测试之前构造一次
    # 因此直接用递归方式实现
    def reconstruct(self):
        structure = []
        history = {}
        old_structure = self.structure
        
        # 定义递归函数
        def dfs(node):
            if node['type'] == 't':
                history_key = node['history_key']
                if history_key in history:
                    idx = history[history_key]
                else:
                    idx = len(structure)
                    history[history_key] = idx
                    structure.append(node)
            else:
                history_key = node['history_key']
                if history_key not in history:
                    temp = []
                    for idy in node['children']:
                        idx = dfs(old_structure[idy])
                        temp.append(idx)
                    node['children'] = temp
                    idx = len(structure)
                    history[history_key] = idx
                    structure.append(node)
                else:
                    idx = history[history_key]         
            return idx
        dfs(old_structure[0])
        return structure
    
    # 前向函数，pytorch中必须实现的函数
    def forward(self, x):

        if not hasattr(self, "structure"):
            self.sub_nums = x.size(1)
            self.structure = self.construct()
            self.structure = self.reconstruct()

        # 首先将input features拆分成不同的word xs
        xs = x.split(1, dim=1)
        bz = x.size(0)
        hidden_size = x.size(-1)
        # datas 用于记录每个节点的输出结果
        datas = []
        for c_node in self.structure:
            if c_node['type'] == 'and':
                left_id = c_node['children'][0]
                right_id = c_node['children'][1]
                # # and 节点，将左右子节点的输出数据concat
                tmp = torch.cat((datas[left_id], datas[right_id]), dim=1)
                if self.config.add_t:
                    tmp, = self.attn(tmp)
                    tmp = self.linear(tmp)
                datas.append(tmp)
            elif c_node['type'] == 'or':
                # # # or 节点，将所有子节点的输出数据相加求和
                #self-attention now use
                tmp = None
                # or 节点，将所有子节点的输出数据相加求和
                for idx in c_node['children']:
                    if tmp is None:
                        tmp = datas[idx]
                    else:
                        tmp = tmp + datas[idx]
                tmp, = self.attn(tmp / len(c_node['children']))
                tmp = self.linear(tmp)
                #new attention
                # weights = torch.zeros([bz, 0], device=x.device)
                # for idx in c_node['children']:
                #     weight = self.q(datas[idx]).sum(dim=1).squeeze().reshape([-1, 1])
                #     weights = torch.concatenate([weights, weight], dim=1)
                # weights = nn.functional.softmax(weights, dim=1)
                # tmp = None
                # for i, idx in enumerate(c_node['children']):
                #     sub_num_blocks = datas[idx].size(1)
                #     if tmp is None:
                #         tmp = datas[idx] * weights[:, i].reshape([bz, 1, 1]).repeat(1,sub_num_blocks,hidden_size)
                #     else:
                #         tmp += datas[idx] * weights[:, i].reshape([bz, 1, 1]).repeat(1,sub_num_blocks,hidden_size)
                # tmp = self.linear(tmp)
                datas.append(tmp)
            else:
                # t 节点，直接将输入xs中对应的部分执行
                if c_node['start'] == c_node['end']:
                    tmp = xs[c_node['start']]
                    if self.config.add_t:
                        tmp = self.linear(tmp)
                    datas.append(tmp)
                else:
                    tmp = torch.cat(xs[c_node['start']:c_node['end']+1], dim=1)
                    if self.config.add_t:
                        tmp, = self.attn(tmp)
                        tmp = self.linear(tmp)
                    datas.append(tmp)
        #最终返回List中最后一个节点的结果，作为整个graph的输出结果
        return datas[-1]




if __name__ == '__main__':
    model = RobertaForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/roberta-base')
    config = model.config
    num_blocks = 32
    aogblock = AOG_Building_Block(num_blocks=num_blocks,config=config).cuda(3)
    optimizer = SGD(aogblock.parameters(),lr=1e-5,momentum=0.9)
    print(len(aogblock.structure))
    print(aogblock.structure[0])
    data = torch.rand([128,num_blocks, config.hidden_size]).cuda(3)
    output = aogblock(data)
    print(output.size())
    loss = output.sum()
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()