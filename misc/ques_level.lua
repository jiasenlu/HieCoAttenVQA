require 'nn'
local utils = require 'misc.utils'
local LSTM = require 'misc.LSTM'
local LanguageEmbedding = require 'misc.LanguageEmbedding'
local attention = require 'misc.attention'
local LanguageEmbedding = require 'misc.LanguageEmbedding'

local layer, parent = torch.class('nn.ques_level', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    
    self.core = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout)
    
    if self.atten_type == 'Alternating' then
      self.atten = attention.alternating_atten(self.hidden_size, self.hidden_size, self.hidden_size, self.seq_length, 196)   
    elseif self.atten_type == 'Parallel' then
      self.atten = attention.parallel_atten(self.hidden_size, self.hidden_size, self.hidden_size, self.seq_length, 196)   
    else
      error('Must provide an valid attention type.')
    end   
    self:_createInitState(1)
    self.mask = torch.Tensor()
    self.core_output = torch.Tensor()
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
  end

function layer:createClones()
    print('constructing clones inside the ques_level')
    self.cores = {self.core}
    for t=1,self.seq_length do
        self.cores[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.cores == nil then self:createClones(); return; end
    print('resharing clones inside the ques_level')
    self.cores[1] = self.core
    for t=1,self.seq_length do
        self.cores[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:getModulesList()
    return {self.core, self.atten}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p1,g1 = self.core:parameters()
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
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:training() end
    self.atten:training()
end

function layer:evaluate()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:evaluate() end
    self.atten:evaluate()
end

function layer:updateOutput(input)
  local ques = input[1]
  local seq_len = input[2]
  local img = input[3]
  self.mask = input[4]


  if self.cores == nil then self:createClones() end -- lazily create clones on first forward pass
  local batch_size = ques:size(1)
  self.tmax = torch.max(seq_len)
  self.tmin = torch.min(seq_len)

  self:_createInitState(batch_size)
  self.fore_state = {[0] = self.init_state}
  self.fore_inputs = {}
  self.core_output:resize(batch_size, self.seq_length, self.rnn_size):zero()


  for t=1,self.tmax do
      self.fore_inputs[t] = {ques:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(self.fore_state[t-1])}
      local out = self.cores[t]:forward(self.fore_inputs[t])
      if t > self.tmin then
        for i=1,self.num_state+1 do
          out[i]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(out[i]), 0)
        end
      end
      self.fore_state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.fore_state[t], out[i]) end
      
      self.core_output:narrow(2,t,1):copy(out[self.num_state+1])
  end
  
  local w_lstm_ques, w_lstm_img, ques_atten, img_atten = unpack(self.atten:forward({self.core_output, img, self.mask}))

  return {w_lstm_ques, w_lstm_img, ques_atten, img_atten}
end

function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local seq_len = input[2]
  local img = input[3]

  local batch_size = ques:size(1)

  local d_core_output, d_imgfeat, dummy = unpack(self.atten:backward({self.core_output, img, self.mask}, gradOutput))

  -- go backwards and lets compute gradients
  local d_core_state = {[self.tmax] = self.init_state} -- initial dstates
  local d_embed_core = d_embed_core or self.core_output:new()
  d_embed_core:resize(batch_size, self.seq_length, self.rnn_size):zero()

  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#d_core_state[t] do table.insert(dout, d_core_state[t][k]) end
    table.insert(dout, d_core_output:narrow(2,t,1):contiguous():view(-1, self.hidden_size))
    local dinputs = self.cores[t]:backward(self.fore_inputs[t], dout)

    if t > self.tmin then
      for k=1,self.num_state+1 do
        dinputs[k]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(dinputs[k]), 0)
      end
    end
    d_core_state[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
    d_embed_core:narrow(2,t,1):copy(dinputs[1])
  end
  self.gradInput = {d_embed_core, d_imgfeat}
  return self.gradInput
end
