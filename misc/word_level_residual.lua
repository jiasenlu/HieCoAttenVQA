require 'nn'
local utils = require 'misc.utils'
local LSTM = require 'misc.LSTM'
local LanguageEmbedding = require 'misc.LanguageEmbedding'
local attention = require 'misc.attention'


-------------------------------------------------------------------------------
-- Language Model core Followed by neualtalk2. 
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.word_level', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
    self.output_size = utils.getopt(opt, 'output_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    -- options for Language Model
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.input_encoding_size, self.input_encoding_size, self.seq_length)    
    
    if self.atten_type == 'interactive' then
      self.atten = attention.inter_atten(512, 512, 512, self.seq_length, 196)   
    elseif self.atten_type == 'synn' then
      self.atten = attention.syn_atten(512, 512, 512, self.seq_length, 196)   
    else
      error('Must provide an valid attention type.')
    end    
    self.cnn = nn.Sequential()
                    :add(nn.View(2048):setNumInputDims(2))
                    :add(nn.Linear(2048, 512))
                    :add(nn.View(196, 512):setNumInputDims(2))
                    :add(nn.Tanh())
                    :add(nn.Dropout(0.5))

    self.mask = torch.Tensor()     
end


function layer:getModulesList()
    return {self.LE, self.atten, self.cnn}
end

function layer:parameters()
    local p2,g2 = self.cnn:parameters()
    local p3,g3 = self.LE:parameters()
    local p4,g4 = self.atten:parameters()

    local params = {}
    for k,v in pairs(p2) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end
    for k,v in pairs(p4) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g2) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end
    for k,v in pairs(g4) do table.insert(grad_params, v) end

    return params, grad_params
end


function layer:training()
    self.LE:training()
    self.atten:training()
    self.cnn:training()
end

function layer:evaluate()
    self.atten:evaluate()
    self.LE:evaluate()
    self.cnn:evaluate()
end

function layer:updateOutput(input)
  local seq = input[1]
  local img = input[2]

  local batch_size = seq:size(1)
  self.mask:resizeAs(seq):zero()
  self.mask[torch.eq(seq, 0)] = 1

  self.img_feat = self.cnn:forward(img)

  self.embed_output = self.LE:forward(seq)
  --local w_embed_ques, w_embed_img = unpack(self.atten:forward({self.embed_output, self.img_feat, self.mask}))
  local w_embed_ques, w_embed_img, ques_atten, img_atten = unpack(self.atten:forward({self.embed_output, self.img_feat, self.mask}))

  --return {self.embed_output, self.img_feat, w_embed_ques, w_embed_img, self.mask}
  return {self.embed_output, self.img_feat, w_embed_ques, w_embed_img, self.mask, ques_atten, img_atten}
end

function layer:updateGradInput(input, gradOutput)
  local seq = input[1]
  local img = input[2]

  local batch_size = seq:size(1)

  local d_embed_ques, d_embed_img, dummy = unpack(self.atten:backward({self.embed_output, self.img_feat, self.mask}, {gradOutput[2], gradOutput[3]}))

  d_embed_ques:add(gradOutput[1])

  local dummy = self.LE:backward(seq, d_embed_ques)

  d_embed_img:add(gradOutput[4])
  d_embed_img:add(gradOutput[5])
  local d_imgfeat = self.cnn:backward(img, d_embed_img)
  self.gradInput = d_imgfeat
  
  return self.gradInput
end