require 'nn'
local utils = require 'misc.utils'
local LSTM = require 'misc.LSTM'
local LanguageEmbedding = require 'misc.LanguageEmbedding'
local attention = require 'misc.attention'


local layer, parent = torch.class('nn.phrase_level', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    self.output_size = utils.getopt(opt, 'output_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    -- options for Language Model
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    
    self.conv = LanguageEmbedding.conv(512, 512, self.seq_length)

    if self.atten_type == 'interactive' then
      self.atten = attention.inter_atten(512, 512, 512, self.seq_length, 196)   
    elseif self.atten_type == 'synn' then
      self.atten = attention.syn_atten(512, 512, 512, self.seq_length, 196)   
    else
      error('Must provide an valid attention type.')
    end      
end

function layer:getModulesList()
    return {self.conv, self.atten}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p1,g1 = self.conv:parameters()
    local p3,g3 = self.atten:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end

    return params, grad_params
end


function layer:training()
    self.atten:training()
    self.conv:training()

end

function layer:evaluate()
    self.atten:evaluate()
    self.conv:evaluate()
end

function layer:updateOutput(input)
  local ques = input[1]
  local seq_len = input[2]
  local img = input[3]
  self.mask = input[4]

  self.conv_out = self.conv:forward(ques)


  local w_conv_ques, w_conv_img, ques_atten, img_atten = unpack(self.atten:forward({self.conv_out, img, self.mask}))
  --local w_lstm1_ques, w_lstm1_img, ques_atten, img_atten, weight = unpack(self.atten:forward({self.conv_out, img, self.mask}))
  return {self.conv_out, w_conv_ques, w_conv_img, ques_atten, img_atten}
  --return {self.core_output, w_lstm1_ques, w_lstm1_img, ques_atten, img_atten, weight}
end

function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local seq_len = input[2]
  local img = input[3]

  local batch_size = ques:size(1)

  local d_core_output, d_imgfeat, dummy = unpack(self.atten:backward({self.conv_out, img, self.mask},  {gradOutput[2], gradOutput[3]}))

  d_core_output:add(gradOutput[1])

  local d_embedding = self.conv:backward(ques, d_core_output)  

  self.gradInput = {d_embedding,d_imgfeat}


  return self.gradInput
end
