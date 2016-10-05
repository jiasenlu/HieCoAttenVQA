require 'nn'
local utils = require 'misc.utils'
local LanguageEmbedding = require 'misc.LanguageEmbedding'
local attention = require 'misc.attention'


local layer, parent = torch.class('nn.word_level', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    self.feature_type = utils.getopt(opt, 'feature_type')
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.hidden_size, self.hidden_size, self.seq_length)    
    
    if self.atten_type == 'Alternating' then
      self.atten = attention.alternating_atten(self.hidden_size, self.hidden_size, self.hidden_size, self.seq_length, 196)   
    elseif self.atten_type == 'Parallel' then
      self.atten = attention.parallel_atten(self.hidden_size, self.hidden_size, self.hidden_size, self.seq_length, 196)   
    else
      error('Must provide an valid attention type.')
    end
    
    if self.feature_type == 'VGG' then
      self.cnn = nn.Sequential()
                      :add(nn.View(512):setNumInputDims(2))
                      :add(nn.Linear(512, self.hidden_size))
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
                      :add(nn.Dropout(0.5))
    elseif self.feature_type == 'Residual' then
      self.cnn = nn.Sequential()
                      :add(nn.View(2048):setNumInputDims(2))
                      :add(nn.Linear(2048, self.hidden_size))
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
                      :add(nn.Dropout(0.5))
    end  
end


function layer:getModulesList()
    return {self.LE, self.atten, self.cnn}
end

function layer:parameters()
    local p1,g1 = self.cnn:parameters()

    local p2,g2 = self.LE:parameters()
    local p3,g3 = self.atten:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p2) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end

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
  self.mask = self.mask or torch.CudaByteTensor()
  self.mask:resize(seq:size()):zero()
  self.mask[torch.eq(seq, 0)] = 1

  self.img_feat = self.cnn:forward(img)

  self.embed_output = self.LE:forward(seq)
  local w_embed_ques, w_embed_img, ques_atten, img_atten = unpack(self.atten:forward({self.embed_output, self.img_feat, self.mask}))

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
